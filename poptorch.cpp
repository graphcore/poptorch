#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/script.h>

void pipeline_stage(int64_t pstage) {
}

void virtual_graph(int64_t vgraph) {}
static auto registry =
    torch::RegisterOperators("poptorch::pipeline_stage", &pipeline_stage)
        .op("poptorch::virtual_graph", &virtual_graph);

std::shared_ptr<torch::jit::Graph> popartMappingPass(std::shared_ptr<torch::jit::Graph> &graph) {
  auto new_graph = std::make_shared<torch::jit::Graph>(graph->current_scope());

  for (auto *node : graph->nodes()) {
    auto kind = node->kind();
    if (std::string(kind.toDisplayString()) == "prim::Constant") {
      new_graph->appendNode(new_graph->create(torch::jit::Symbol::fromQualString("popart::Constant"), 1));
    }
  }

  return new_graph;
}

void transformPass(std::string model_path, std::vector<std::pair<std::string, std::vector<int64_t>>> inputs) {
  std::cout << "Running transform pass on " << model_path << "\n";

  torch::jit::script::Module module;
  try {
    module = torch::jit::load(model_path);
  } catch (const c10::Error &e) {
    std::cerr << "Error loading '" << model_path << "'\n";
    return;
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
    popartMappingPass(graph)->dump();
  }

  std::cout << "exiting transformPass\n";
}


PYBIND11_MODULE(poptorch_core, m) { m.def("transformPass", transformPass);
}
