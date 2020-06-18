#include <poptorch/Peephole.hpp>

namespace poptorch {

class PeepholeOptimizer {
public:
  PeepholeOptimizer(bool _training) : isTraining(_training) {}

  void run(torch::jit::Graph &graph) {
    run(graph.block());

    for (auto node : toDelete) {
      node->destroy();
    }
  }

private:
  void removeUncheckedCast(torch::jit::Node *node) {
    assert(node->kind() == c10::prim::unchecked_cast);
    node->output()->replaceAllUsesWith(node->input());
    markAsDelete(node);
  }

  void removeNodeWithoutOutput(torch::jit::Node *node) {
    assert(node->outputs().size() == 0);
    markAsDelete(node);
  }

  void handleGetAttrNode(torch::jit::Node *node) {
    assert(node->kind() == c10::prim::GetAttr);
    if (node->s(c10::Symbol::fromQualString("attr::name")) == "training") {
      auto graph = node->owningGraph();
      graph->setInsertPoint(node);
      torch::jit::Value *newConst =
          graph->insertConstant(isTraining, node->sourceRange(), node->scope());
      node->output()->replaceAllUsesWith(newConst);
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
    default:
      break;
    }
  }

  void run(torch::jit::Block *block) {
    for (auto node : block->nodes()) {
      for (auto b : node->blocks()) {
        run(b);
      }

      run(node);
    }
  }

  void markAsDelete(torch::jit::Node *node) { toDelete.insert(node); }

  bool isTraining;
  std::unordered_set<torch::jit::Node *> toDelete;
};

void peepholeOptimizations(torch::jit::Graph &graph, bool training) {
  PeepholeOptimizer(training).run(graph);
}

} // namespace poptorch
