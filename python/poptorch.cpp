#include <iostream>


#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/script.h>

#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/pybind_utils.h>

#include "poptorch/LowerToPopart.hpp"

void pipeline_stage(int64_t pstage) {}

void virtual_graph(int64_t vgraph) {}
static auto registry =
    torch::RegisterOperators("poptorch::pipeline_stage", &pipeline_stage)
        .op("poptorch::virtual_graph", &virtual_graph);

namespace poptorch {

void popartMappingPass(std::shared_ptr<torch::jit::Graph> &graph) {
  for (torch::jit::Node *node : graph->nodes()) {

    torch::jit::Symbol kind = node->kind();
    if (std::string(kind.toDisplayString()) == "prim::Constant") {
      /*graph->appendNode(new_graph->create(
          torch::jit::Symbol::fromQualString("popart::Constant"), 1));*/
    }
  }
}

pybind11::object traceAndRun(
    std::string model_path, pybind11::tuple inputs) {
  std::cout << "Running transform pass on " << model_path << "\n";

  torch::jit::script::Module module;
  try {
    module = torch::jit::load(model_path);
  } catch (const c10::Error &e) {
    std::cerr << "Error loading '" << model_path << "'\n";
    return {};
  }

  auto forward = module.get_method("forward");
  {
    std::shared_ptr<torch::jit::Graph> graph = forward.graph();
    std::cout << "Dumping graph:\n";
    graph->dump();
  }

  {
    auto graphAndTensors =
        torch::jit::LowerGraph(*forward.graph(), module._ivalue());
    std::cout << "Dumping lowered graph:\n";
    auto graph = graphAndTensors.first;
    graph->dump();

    torch::jit::RemoveInplaceOps(graph);

    popartMappingPass(graph);

    // Create a jit stack from the incoming pytorch tensors.
    torch::jit::Stack inputStack = torch::jit::toTraceableStack(inputs);

    // And turn convert them into at tensors which we can then resolve the address of.
    std::vector<at::Tensor> inputTensors;

    for (torch::jit::IValue value : inputStack) {
      inputTensors.push_back(value.toTensor());
    // .is_contiguous() << "  Data: " << value.toTensor().data_ptr() << std::endl;
    }

    at::IValue tensor = poptorch::lowerToPopart(*graph, inputTensors);
    graph->dump();

    return torch::jit::toPyObject(tensor);
  }

  std::cout << "exiting transformPass\n";
}

} // namespace poptorch


PYBIND11_MODULE(poptorch_core, m) {
  m.def("traceAndRun", poptorch::traceAndRun);
}
