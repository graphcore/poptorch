// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include "poptorch/EliminateListConstructs.hpp"
#include "poptorch/LowerToPopart.hpp"
#include "poptorch/Peephole.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/ShapeInference.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/script.h>

void begin_ipu_block(int64_t ipu_id) { UNUSED(ipu_id); }
void end_ipu_block() {}

at::Tensor ipu_print_tensor(at::Tensor t) { return t; }
at::Tensor identity_loss(at::Tensor t, int64_t reduction) {
  UNUSED(reduction);
  return t;
}

static auto registry =
    torch::RegisterOperators("poptorch::begin_ipu_block", &begin_ipu_block)
        .op("poptorch::end_ipu_block", &end_ipu_block)
        .op("poptorch::ipu_print_tensor", &ipu_print_tensor)
        .op("poptorch::identity_loss", &identity_loss);
//.op("popart::convolution", convolution,
// torch::RegisterOperators::options().aliasAnalysis(c10::AliasAnalysisKind::INTERNAL_SPECIAL_CASE));

namespace poptorch {

std::vector<pybind11::object>
execute(std::shared_ptr<poptorch::PoplarExecutable> executable,
        pybind11::tuple inputs) {

  // Create a jit stack from the incoming pytorch tensors.
  torch::jit::Stack inputStack = torch::jit::toTraceableStack(inputs);

  // And turn convert them into at tensors which we can then resolve the
  // address of.
  std::vector<at::Tensor> inputTensors;
  for (torch::jit::IValue value : inputStack) {
    inputTensors.push_back(value.toTensor());
  }

  std::vector<at::IValue> value = executable->Run(inputTensors);

  std::vector<pybind11::object> returnee;
  std::transform(value.begin(), value.end(), std::back_inserter(returnee),
                 [](at::IValue &v) { return torch::jit::toPyObject(v); });

  return returnee;
}

torch::jit::script::Module *as_module(py::handle h) {
  return reinterpret_cast<torch::jit::script::Module *>(
      pybind11::detail::values_and_holders(
          reinterpret_cast<pybind11::detail::instance *>(h.ptr()))
          .begin()
          ->value_ptr());
}

torch::jit::Graph *as_graph(py::handle h) {
  return reinterpret_cast<torch::jit::Graph *>(
      pybind11::detail::values_and_holders(
          reinterpret_cast<pybind11::detail::instance *>(h.ptr()))
          .begin()
          ->value_ptr());
}

void constantPropagation(torch::jit::Graph *graph) {
  // Create a shared_ptr with a custom deleter that doesn't do anything.
  std::shared_ptr<torch::jit::Graph> x(graph, [](torch::jit::Graph *) {});
  ERROR_ON_MSG(x.use_count() != 1, "x should be the only handle to graph");
  torch::jit::ConstantPropagation(x);
  ERROR_ON_MSG(x.use_count() != 1, "x should be the only handle to graph");
}

std::shared_ptr<poptorch::PoplarExecutable>
compileWithTrace(py::handle h, py::handle, pybind11::tuple inputs,
                 std::uint64_t steps, bool training,
                 std::uint64_t replicationFactor,
                 std::uint64_t gradientAccumulation, bool profile) {
  auto module = as_module(h);

  auto forward = module->get_method("forward");
  auto graphAndTensors =
      torch::jit::LowerGraph(*forward.graph(), module->_ivalue());
  auto graph = graphAndTensors.first;

  torch::jit::EliminateDeadCode(graph);
  torch::jit::PeepholeOptimize(graph);
  torch::jit::EliminateDeadCode(graph);

  torch::jit::RemoveInplaceOps(graph);

  logging::debug("Graph right before canonicalization:\n{}", *graph);

  poptorch::CanonicalizeLists(*graph);

  // Convert any unsupported ATEN nodes in the graph to a popart
  // representation.
  poptorch::Canonicalize(*graph);

  // Enforce any constraints that aren't enforced by popart.
  poptorch::CanonicalizeLate(*graph);

  // Create a jit stack from the incoming pytorch tensors.
  torch::jit::Stack inputStack = torch::jit::toTraceableStack(inputs);

  // And turn convert them into at tensors which we can then resolve the
  // address of.
  std::vector<at::Tensor> inputTensors;
  for (torch::jit::IValue value : inputStack) {
    inputTensors.push_back(value.toTensor());
  }

  // Find the parameter data from.
  std::vector<at::Tensor> parameterData;
  for (at::Tensor param : graphAndTensors.second) {
    parameterData.push_back(param);
  }

  logging::debug("Graph right before popart:\n{}", *graph);

  return poptorch::lowerToPopart(*graph, inputTensors, parameterData, steps,
                                 training, replicationFactor,
                                 gradientAccumulation, profile);
}

std::shared_ptr<poptorch::PoplarExecutable>
compileWithScript(py::handle h, py::handle g, pybind11::tuple inputs,
                  std::uint64_t steps, bool training,
                  std::uint64_t replicationFactor,
                  std::uint64_t gradientAccumulation, bool profile) {
  auto module = as_module(h);
  auto argGraph = as_graph(g);

  torch::jit::Inline(*argGraph);
  constantPropagation(argGraph);
  peepholeOptimizations(*argGraph, training);

  auto graphAndTensors = torch::jit::LowerGraph(*argGraph, module->_ivalue());
  auto graph = graphAndTensors.first;
  graph->dump();

  int loop_count = 0;
  std::string graphString;
  while (true) {
    propagateInputShapes(graph.get());
    torch::jit::PeepholeOptimize(graph, true);
    torch::jit::ConstantPropagation(graph);
    torch::jit::EliminateDeadCode(graph);
    peepholeOptimizations(*graph, training);

    std::string postPassesGraph = graph->toString(false);
    if (graphString == postPassesGraph) {
      std::cout << "Breaking from const folding after " << loop_count
                << " iterations.\n";
      break;
    } else {
      graphString = std::move(postPassesGraph);
    }

    loop_count++;
  }

  torch::jit::RemoveInplaceOps(graph);

  logging::debug("Graph right before canonicalization:\n{}", *graph);

  // Convert any unsupported ATEN nodes in the graph to a popart
  // representation.
  poptorch::Canonicalize(*graph);

  // Clean up the module as we will likely have stopped using lots of
  // constants.
  // TODO: Alias Analysis freaks out about this, so ignore for now.
  // torch::jit::EliminateDeadCode(graph);

  // Create a jit stack from the incoming pytorch tensors.
  torch::jit::Stack inputStack = torch::jit::toTraceableStack(inputs);

  // And turn convert them into at tensors which we can then resolve the
  // address of.
  std::vector<at::Tensor> inputTensors;
  for (torch::jit::IValue value : inputStack) {
    inputTensors.push_back(value.toTensor());
  }

  // Find the parameter data from.
  std::vector<at::Tensor> parameterData = graphAndTensors.second;
  std::cout << "There should be " << parameterData.size() << " parameters.\n";

  logging::debug("Graph right before popart:\n{}", *graph);

  return poptorch::lowerToPopart(*graph, inputTensors, parameterData, steps,
                                 training, replicationFactor,
                                 gradientAccumulation, profile);
}

void pyPropagateInputShapes(py::handle h) {
  auto graph = as_graph(h);
  propagateInputShapes(graph);
}

void pyPeepholeOptimizations(py::handle h, bool training) {
  auto graph = as_graph(h);
  peepholeOptimizations(*graph, training);
}

void pyEliminateListConstructs(py::handle h) {
  auto graph = as_graph(h);
  eliminateListConstructs(graph);
}

void pyCanonicalize(py::handle h) {
  auto graph = as_graph(h);
  Canonicalize(*graph);
}

} // namespace poptorch

PYBIND11_MODULE(poptorch_core, m) {
  py::class_<poptorch::PoplarExecutable,
             std::shared_ptr<poptorch::PoplarExecutable>>(
      m, "InternalPoplarExecutable");

  m.def("compileWithTrace", poptorch::compileWithTrace);
  m.def("compileWithScript", poptorch::compileWithScript);
  m.def("execute", poptorch::execute);
  m.def("propagateInputShapes", poptorch::pyPropagateInputShapes);
  m.def("peepholeOptimizations", poptorch::pyPeepholeOptimizations);
  m.def("eliminateListConstructs", poptorch::pyEliminateListConstructs);
  m.def("canonicalize", poptorch::pyCanonicalize);
}
