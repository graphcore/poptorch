// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poptorch/Peephole.hpp>

#include <poptorch/Utils.hpp>
#include <poptorch_logging/Error.hpp>

namespace poptorch {

class PeepholeOptimizer {
public:
  explicit PeepholeOptimizer(bool _training) : _is_training(_training) {}

  void run(torch::jit::Graph *graph) {
    run(graph->block());

    for (auto node : _to_delete) {
      node->destroy();
    }
  }

private:
  void removeUncheckedCast(torch::jit::Node *node) {
    ERROR_ON(node->kind() != c10::prim::unchecked_cast);
    node->output()->replaceAllUsesWith(node->input());
    markAsDelete(node);
  }

  void removeNodeWithoutOutput(torch::jit::Node *node) {
    ERROR_ON(!node->outputs().empty());
    markAsDelete(node);
  }

  void handleGetAttrNode(torch::jit::Node *node) {
    ERROR_ON(node->kind() != c10::prim::GetAttr);
    if (node->s(c10::attr::name) == "training") {
      auto graph = node->owningGraph();
      graph->setInsertPoint(node);
      torch::jit::Value *new_const = graph->insertConstant(
          _is_training, node->sourceRange(), node->scope());
      node->output()->replaceAllUsesWith(new_const);
      markAsDelete(node);
    }
  }

  void run(torch::jit::Node *node) {
    auto kind = node->kind();

    switch (kind) {
    case c10::prim::unchecked_cast:
      removeUncheckedCast(node);
      break;
    case c10::prim::RaiseException:
    case c10::prim::SetAttr:
      removeNodeWithoutOutput(node);
      break;
    case c10::prim::GetAttr:
      handleGetAttrNode(node);
      break;
    default:
      break;
    }
  }

  void run(torch::jit::Block *block) {
    for (auto node : block->nodes()) {
      logging::LogContext ctx("PeepholeOptimizer Processing " +
                              nodeToString(node));
      for (auto b : node->blocks()) {
        run(b);
      }

      run(node);
    }
  }

  void markAsDelete(torch::jit::Node *node) { _to_delete.insert(node); }

  bool _is_training;
  std::unordered_set<torch::jit::Node *> _to_delete;
};

void peepholeOptimizations(torch::jit::Graph *graph, bool training) {
  PeepholeOptimizer(training).run(graph);
}

} // namespace poptorch
