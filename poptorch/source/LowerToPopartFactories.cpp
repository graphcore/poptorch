// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "poptorch/LowerToPopartFactories.hpp"

#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/peephole.h>
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

  poptorch::fuseScatters(graph.get());

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

  lower.lower();

  return lower;
}

} // namespace poptorch
