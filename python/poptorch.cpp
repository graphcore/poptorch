#include <iostream>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/script.h>

#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include "shared/Logging.hpp"
#include "poptorch/LowerToPopart.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Peephole.hpp"
#include "poptorch/ShapeInference.hpp"

void begin_ipu_block(int64_t ipu_id) {}
void end_ipu_block() { }

at::Tensor ipu_print_tensor(at::Tensor t) {
  return t;
}

static auto registry =
    torch::RegisterOperators("poptorch::begin_ipu_block", &begin_ipu_block)
        .op("poptorch::end_ipu_block", &end_ipu_block)
        .op("poptorch::ipu_print_tensor", &ipu_print_tensor);
        //.op("popart::convolution", convolution, torch::RegisterOperators::options().aliasAnalysis(c10::AliasAnalysisKind::INTERNAL_SPECIAL_CASE));

namespace poptorch {


pybind11::object execute(std::shared_ptr<poptorch::PoplarExecutable> executable, pybind11::tuple inputs) {

  // Create a jit stack from the incoming pytorch tensors.
  torch::jit::Stack inputStack = torch::jit::toTraceableStack(inputs);

  // And turn convert them into at tensors which we can then resolve the address of.
  std::vector<at::Tensor> inputTensors;
  for (torch::jit::IValue value : inputStack) {
    inputTensors.push_back(value.toTensor());
  }

  at::IValue value = executable->Run(inputTensors);
  return torch::jit::toPyObject(value);
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

std::shared_ptr<poptorch::PoplarExecutable> compile(
  py::handle h, pybind11::tuple inputs, std::uint64_t steps, bool training, std::uint64_t replicationFactor,  std::uint64_t gradientAccumulation) {
  auto module = as_module(h);

  auto forward = module->get_method("forward");
  auto graphAndTensors =
      torch::jit::LowerGraph(*forward.graph(), module->_ivalue());
  auto graph = graphAndTensors.first;


  torch::jit::RemoveInplaceOps(graph);
  // Convert any unsupported ATEN nodes in the graph to a popart representation.
  poptorch::Canonicalize(*graph);

  // Clean up the module as we will likely have stopped using lots of constants.
  // TODO: Alias Analysis freaks out about this, so ignore for now.
  //torch::jit::EliminateDeadCode(graph);

  // Create a jit stack from the incoming pytorch tensors.
  torch::jit::Stack inputStack = torch::jit::toTraceableStack(inputs);

  // And turn convert them into at tensors which we can then resolve the address of.
  std::vector<at::Tensor> inputTensors;
  for (torch::jit::IValue value : inputStack) {
    inputTensors.push_back(value.toTensor());
  }

  // Find the parameter data from.
  std::vector<at::Tensor> parameterData;
  for (at::Tensor param :graphAndTensors.second) {
    parameterData.push_back(param);
  }

  logging::debug("Graph right before popart:\n{}", *graph);

  return poptorch::lowerToPopart(*graph, inputTensors, parameterData, steps, training, replicationFactor, gradientAccumulation);
}

void pyPropagateInputShapes(py::handle h) {
  auto graph = as_graph(h);
  propagateInputShapes(graph);
}

void pyPeepholeOptimizations(py::handle h, bool training) {
  auto graph = as_graph(h);
  peepholeOptimizations(*graph, training);
}

} // namespace poptorch


PYBIND11_MODULE(poptorch_core, m) {
  py::class_<poptorch::PoplarExecutable, std::shared_ptr<poptorch::PoplarExecutable> >(m, "InternalPoplarExecutable");

  m.def("compile", poptorch::compile);
  m.def("execute", poptorch::execute);
  m.def("propagateInputShapes", poptorch::pyPropagateInputShapes);
  m.def("peepholeOptimizations", poptorch::pyPeepholeOptimizations);
}
