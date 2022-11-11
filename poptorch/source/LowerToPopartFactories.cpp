// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "poptorch/LowerToPopartFactories.hpp"

#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/script.h>

#include "popart_compiler/Compiler.hpp"
#include "poptorch_logging/Logging.hpp"
#include "poptorch_logging/Tracepoint.hpp"

#include "poptorch/AliasProcessing.hpp"
#include "poptorch/DispatchTracer.hpp"
#include "poptorch/ImplicitCasting.hpp"
#include "poptorch/InplaceOps.hpp"
#include "poptorch/OverlappedIO.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/RequiresGrad.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"

namespace poptorch {

namespace {
// Prints the graph but substitutes BFloat16 for Float16/Float32
void printGraphBeforeHalfFloatResolution(const torch::jit::Graph &graph) {
  std::ostringstream graph_oss;
  graph_oss << graph;
  std::string graph_str = graph_oss.str();

  size_t start = 0;
  const std::string from = "BFloat16";
  const std::string to = "Float16/Float32";
  while ((start = graph_str.find(from, start)) != std::string::npos) {
    graph_str.replace(start, from.length(), to);
    start += to.length();
  }

  logging::trace("Graph right before half/float resolution:\n{}", graph_str);
}

// Recursively print the input type, which may be:
// - A tensor
// - A tuple (Which may contain tensors or tuples)
template <typename TensorIt>
void printOrigType(std::ostream &os, const c10::TypePtr &type,
                   TensorIt &current) {
  auto tuple_type = type->cast<c10::TupleType>();
  if (tuple_type != nullptr) {
    // Print and recurse through tuples
    os << '(';
    for (std::size_t i = 0; i < tuple_type->containedTypes().size(); i++) {
      auto t = tuple_type->containedTypes()[i];
      printOrigType(os, t, current);
      if (i != tuple_type->containedTypes().size() - 1) {
        os << ", ";
      }
    }
    os << ')';
    return;
  }
  auto tensor_type = type->cast<c10::TensorType>();
  if (tensor_type != nullptr) {
    // Recursion base case, just print the original tensor type
    os << *tensor_type->withScalarType(current->scalar_type());
    // Advance the current tensor iterator so that the next invocation
    // uses the correct tensor
    std::advance(current, 1);
    return;
  }
  ERROR("Invalid type being traced, expected tensors or tuples");
}
// Print the graph input string which matches that of Graph::print
void printGraphInputStr(std::ostream &os, const torch::jit::Graph &graph,
                        const std::vector<at::Tensor> &input_tensors,
                        bool is_trace_input) {
  bool first = true;
  auto it_tensors = input_tensors.begin();
  for (const auto *input : graph.inputs()) {
    if (!first) {
      os << ",\n      ";
    }
    first = false;
    os << "%" << input->debugName();
    os << " : ";
    // After reaching end(input_tensors), the remaining
    // (N - size(input_tensors)) graph inputs will have the same traced
    // type so we don't need to process those
    if (is_trace_input || it_tensors == input_tensors.end()) {
      os << *input->type();
    } else {
      printOrigType(os, input->type(), it_tensors);
    }
  }
}

// Prints the graph out to the log (trace), print both the trace inputs and
// actual inputs if trace_input_str is not empty
void logGraph(const char *intro_str, const torch::jit::Graph &graph,
              bool has_converted_any_half,
              const std::vector<at::Tensor> &input_tensors) {
  std::ostringstream graph_str;
  graph_str << intro_str << "\n";

  // If there are no halves converted to floats, simply print the graph
  if (!has_converted_any_half) {
    graph_str << graph;
    logging::trace("{}", graph_str.str());
    return;
  }

  // Print the trace inputs
  graph_str << "graph(";
  printGraphInputStr(graph_str, graph, input_tensors, true);
  graph_str << "):\n";

  // Print the original inputs
  graph_str << "[orig:";
  printGraphInputStr(graph_str, graph, input_tensors, false);
  graph_str << "]\n";

  std::vector<const torch::jit::Node *> groups;
  for (const auto *n : graph.nodes()) {
    n->print(graph_str, 1, &groups, true);
  }
  graph_str << "  return (" << graph.outputs() << ")\n";
  size_t i = 0;

  for (const auto *fg : groups) {
    graph_str << "with " << fg->kind().toQualString() << "_" << i++ << " = "
              << *fg->g(torch::jit::attr::Subgraph);
  }
  logging::trace("{}", graph_str.str());
}
} // namespace

poptorch::LowerToPopart lowerToPopartFromTrace(
    SessionOptionsParser &parser,
    const std::shared_ptr<torch::jit::Graph> &graph,
    bool has_converted_any_half, bool training,
    std::vector<at::Tensor> &input_tensors,
    std::vector<std::string> &parameters,
    std::vector<at::Tensor> &traced_parameter_tensors,
    AnchorList &&anchors_list, const std::function<void()> &initCallbackBuffers,
    std::vector<popart_compiler::Optimizer> &&optimizers,
    const AttributeAccessor &attribute_accessor, CPUCallbackMap &callbacks) {
  auto &parsed_options = parser.options();
  const char *poptorch_passes = "poptorchPasses";
  const char *lower_to_popart = "lowerToPopart";

  poptorch::logging::Tracepoint::begin(poptorch_passes);

  logGraph("Initial traced graph:", *graph, has_converted_any_half,
           input_tensors);

  torch::jit::EliminateDeadCode(graph);
  torch::jit::PeepholeOptimize(graph);
  torch::jit::LowerSimpleTuples(graph);
  torch::jit::PeepholeOptimize(graph);
  logGraph("Optimised graph:", *graph, has_converted_any_half, input_tensors);

  poptorch::attributiseOverlappedIO(graph.get());
  logGraph("Graph after attributising IO overlap specifiers", *graph,
           has_converted_any_half, input_tensors);

  poptorch::resolveAliases(graph.get());
  logGraph("Graph after resolving aliases:", *graph, has_converted_any_half,
           input_tensors);

  // Ensure that list elements have the ListTypeWithNumElements type which
  // contains the number of elements in a list
  poptorch::type_and_constant_canonicalization::addListNumElements(graph.get());

  InplaceGraphInfo inplace_info = handleInplaceOpsInGraph(
      *graph, traced_parameter_tensors.size(), anchors_list.size(),
      parsed_options.hasInputReplication() &&
          parsed_options.broadcastBuffers());
  logGraph("Graph after handling inplace ops:", *graph, has_converted_any_half,
           input_tensors);

  // Any types with ListTypeWithNumElements must be reverted (revert = true)
  // to allow constant evaluation to proceed
  poptorch::type_and_constant_canonicalization::addListNumElements(graph.get(),
                                                                   true);

  poptorch::type_and_constant_canonicalization::evaluateConstexprs(graph.get());
  logGraph("Graph after evaluating constant expressions:", *graph,
           has_converted_any_half, input_tensors);

  poptorch::type_and_constant_canonicalization::makeConstantIntParams(
      graph.get(), parameters, traced_parameter_tensors);
  logGraph("Graph after casting making integer params as constant "
           "inputs:",
           *graph, has_converted_any_half, input_tensors);

  poptorch::type_and_constant_canonicalization::castUnsupportedInputs(
      graph.get());
  logGraph("Graph after casting unsupported inputs:", *graph,
           has_converted_any_half, input_tensors);

  poptorch::type_and_constant_canonicalization::checkAndChangeOutputTypes(
      graph.get());
  logGraph("Graph after output type changes:", *graph, has_converted_any_half,
           input_tensors);

  std::vector<std::size_t> input_index_map;
  poptorch::type_and_constant_canonicalization::canonicaliseConstants(
      graph.get(), input_index_map);
  logGraph("Graph after constant canonicalisation:", *graph,
           has_converted_any_half, input_tensors);

  // After constant canonicalisation, it is safe to ensure that list have
  // ListTypeWithNumElements once again, plus the last step will have added
  // new list constructs.
  poptorch::type_and_constant_canonicalization::addListNumElements(graph.get());

  // Convert the IR to half to match the inputs/actual usage.
  poptorch::canonicaliseHalfInputs(graph.get(), input_tensors,
                                   traced_parameter_tensors);
  logGraph("Graph after canonicalising half:", *graph, has_converted_any_half,
           input_tensors);

  poptorch::canonicalizeLists(graph.get());
  logging::trace("Graph after canonicalizing lists:\n{}", *graph);

  poptorch::cpuOffloadingCleanup(graph.get());

  poptorch::removeScatterAddIndexExpansion(graph.get());

  poptorch::simplifyGatherWithExpandedIndices(graph.get());

  logging::trace("Graph before PopART canonicalisation:\n{}", *graph);
  // Convert any unsupported ATEN nodes in the graph to a popart
  // representation.
  poptorch::canonicalize(graph.get());

  printGraphBeforeHalfFloatResolution(*graph);

  poptorch::annotateSubgraphs(graph.get(), graph->nodes().front());

  poptorch::resolveHalfOrFloat(graph.get());

  // Enforce any constraints that aren't enforced by popart.
  poptorch::canonicalizeLate(graph.get());
  logging::trace("Graph after PopART canonicalisation:\n{}", *graph);

  if (training) {
    poptorch::removeSurplusIdentityLosses(graph.get());
    poptorch::addDetachOperations(graph.get());
    logging::trace("Graph after adding detach operations:\n{}", *graph);
  }

  // Error the user if any operations couldn't be canonicalised.
  poptorch::errorOnUnsupportedAten(graph.get());

  // Get the callback buffers from python, we have to do this at the last
  // possible moment due to tracing.
  initCallbackBuffers();

  logging::trace("Graph before lowering to PopART:\n{}", *graph);
  poptorch::logging::Tracepoint::end(poptorch_passes);
  poptorch::logging::Tracepoint::begin(lower_to_popart);

  poptorch::LowerToPopart lower(
      graph.get(), traced_parameter_tensors, parameters,
      std::move(inplace_info), training, std::move(optimizers), parsed_options,
      attribute_accessor, callbacks, std::move(anchors_list),
      std::move(input_index_map));
  lower.lower(&input_tensors);

  poptorch::setAvailableMemoryOnGraphFinalized();

  poptorch::logging::Tracepoint::end(lower_to_popart);
  return lower;
}

poptorch::LowerToPopart lowerToPopartFromDispatch(
    SessionOptionsParser &parser, bool training, AnchorList &&anchors_list,
    const std::function<void()> &initCallbackBuffers,
    std::vector<popart_compiler::Optimizer> &&optimizers,
    const AttributeAccessor &attribute_accessor, CPUCallbackMap &callbacks) {
  auto &parsed_options = parser.options();
  const std::shared_ptr<torch::jit::Graph> graph = getTracedGraph();
  logging::trace("Initial dispatched graph:\n{}", *graph);

  fixRequiresGradFromDispatch(graph.get());
  torch::jit::EliminateDeadCode(graph);
  torch::jit::PeepholeOptimize(graph);
  logging::trace("Optimised graph:\n{}", *graph);

  InplaceGraphInfo inplace_info = getInplaceGraphInfo(
      anchors_list.size(), parsed_options.hasInputReplication() &&
                               parsed_options.broadcastBuffers());
  logging::trace("Graph after handling inplace ops:\n{}", *graph);

  poptorch::attributiseOverlappedIO(graph.get());
  logging::trace("Graph after attributising IO overlap specifiers:\n{}",
                 *graph);

  fixForLoopInputs(*graph);

  poptorch::type_and_constant_canonicalization::evaluateConstexprs(graph.get());
  logging::trace("Graph after evaluating constant expressions:\n{}", *graph);

  std::vector<std::size_t> input_index_map;
  poptorch::type_and_constant_canonicalization::canonicaliseConstants(
      graph.get(), input_index_map);
  logging::trace("Graph after constant canonicalisation:\n{}", *graph);

  poptorch::removeScatterAddIndexExpansion(graph.get());

  poptorch::simplifyGatherWithExpandedIndices(graph.get());

  logging::trace("Graph before PopART canonicalisation:\n{}", *graph);
  poptorch::canonicalize(graph.get());

  poptorch::annotateSubgraphs(graph.get(), graph->nodes().front());

  // Collapse any `begin_cpu ... end_cpu` sequences into a single node, with the
  // correct inputs & outputs.
  poptorch::cpuOffloadingCleanup(graph.get());

  if (graph->outputs().empty()) {
    logging::trace("No outputs, so all nodes cleared");
    for (auto it = graph->nodes().rbegin(); it != graph->nodes().rend(); it++) {
      it.destroyCurrent();
    }
  }

  // TODO(T67295): remove after we use our own dispatch key.
  removeDeadImplicitCasts(graph.get());

  canonicalizeLate(graph.get());
  logging::trace("Graph after PopART canonicalisation:\n{}", *graph);

  if (training) {
    poptorch::addDetachOperations(graph.get());
    poptorch::removeSurplusIdentityLosses(graph.get());
    logging::trace("Graph after adding detach operations:\n{}", *graph);
  }

  // Error the user if any operations couldn't be canonicalised.
  poptorch::errorOnUnsupportedAten(graph.get());

  // Prepare CPU op callbacks, by allocating the CPU tensors where the
  // inputs/outputs will be stored. We have to do this at the last possible
  // moment due to tracing.
  initCallbackBuffers();

  logging::trace("Graph before lowering to PopART:\n{}", *graph);
  poptorch::LowerToPopart lower(
      graph.get(), std::move(inplace_info), training, std::move(optimizers),
      parsed_options, attribute_accessor, callbacks, std::move(anchors_list),
      std::move(input_index_map));

  lower.lower(nullptr);

  return lower;
}

} // namespace poptorch
