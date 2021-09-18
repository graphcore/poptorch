// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"

namespace poptorch {
namespace type_and_constant_canonicalization {
namespace {

size_t numNodesInGraph(const torch::jit::Graph *g) {
  return std::distance(g->nodes().begin(), g->nodes().end());
}

size_t numValuesInGraph(const torch::jit::Graph *g) {
  size_t num_values = 0;
  for (auto node : g->nodes()) {
    num_values += node->outputs().size();
  }
  return num_values;
}

class ConstExprEvaluator {
public:
  explicit ConstExprEvaluator(torch::jit::Graph *g)
      : _graph(g), _nodes_map(numNodesInGraph(g)),
        _values_map(numValuesInGraph(g)) {}
  // Guarantees no re-hashing: does not matter if the hash map is sparse

  void evaluate();

private:
  void copyAllConstNodesToToConstexprGraph();

  void removeLoneConstants();

  void evalutateConstExprGraph(torch::jit::Stack *stack);

  void replaceWithConstants(const torch::jit::Stack &stack);

  void removeUnusedNodes();

  bool nodeIsConstExpr(const torch::jit::Node &node) const;

  void copyNodeToConstexprGraph(torch::jit::Node *node);

  static void setAmbiguousValuesToFloatOrHalf(torch::jit::Value *value);

  // Original graph
  torch::jit::Graph *_graph;

  // Graph containing constant expressions which can be evaluated
  std::shared_ptr<torch::jit::Graph> _constexpr_graph;

  // Map the nodes and inputs between the two graphs
  // original -> constexpr
  std::unordered_map<const torch::jit::Node *, torch::jit::Node *> _nodes_map;
  std::unordered_map<const torch::jit::Value *, torch::jit::Value *>
      _values_map;

  // Keep a list of the values in the original graph to be replaced by constants
  std::unordered_set<torch::jit::Value *> _ins_to_make_consts;
};

void ConstExprEvaluator::evaluate() {
  ERROR_ON_MSG(_constexpr_graph,
               "ConstExprEvaluator::evaluate should only be run once");
  _constexpr_graph = std::make_shared<torch::jit::Graph>();

  // Copy all nodes which can be evaluated as a constant expression into a new
  // graph. In addition, set outputs of the new graph where required
  copyAllConstNodesToToConstexprGraph();

  // We do not want to evaluate lone constants only to replace them with an
  // identical constants
  removeLoneConstants();

  // Evaluate the constexpr graph saving the outputs to stack
  torch::jit::Stack stack;
  evalutateConstExprGraph(&stack);

  // Replace outputs in the original graph, with the constants calculated from
  // the constexpr graph
  replaceWithConstants(stack);

  // Remove nodes which are now unused, in the original graph
  removeUnusedNodes();
}

void ConstExprEvaluator::copyAllConstNodesToToConstexprGraph() {
  logging::LogContext ctx_func("ConstExprEvaluator");
  std::vector<torch::jit::Node *> nodes_plus_return;
  for (auto node : _graph->nodes()) {
    nodes_plus_return.push_back(node);
  }
  nodes_plus_return.push_back(_graph->return_node());

  for (auto node : nodes_plus_return) {
    logging::LogContext ctx("processing " + nodeToString(node));

    if (nodeIsConstExpr(*node)) {
      copyNodeToConstexprGraph(node);
    } else {
      for (auto input : node->inputs()) {
        // Add any outputs to the const expression graph
        if (_values_map.count(input) == 1 &&
            _ins_to_make_consts.count(input) == 0) {
          _ins_to_make_consts.emplace(input);
          _constexpr_graph->registerOutput(_values_map[input]);
        }
      }
    }
  }
  logging::trace("Constexpr graph: {}", *_constexpr_graph);
}

void ConstExprEvaluator::removeLoneConstants() {
  for (auto node : _graph->nodes()) {
    if (!node->inputs().empty()) {
      continue;
    }

    if (node->outputs().size() != 1) {
      continue;
    }

    auto new_node = _nodes_map[node];
    auto uses = new_node->output()->uses();
    if (uses.size() != 1) {
      continue;
    }

    if (uses[0].user != _constexpr_graph->return_node()) {
      continue;
    }

    // The node is on its own in the consextpr graph and there is no point
    // replacing it with another single node
    _constexpr_graph->eraseOutput(uses[0].offset);
    new_node->destroy();

    _nodes_map.erase(node);
    _values_map.erase(node->output());
    _ins_to_make_consts.erase(node->output());
  }
}

void ConstExprEvaluator::evalutateConstExprGraph(torch::jit::Stack *stack) {
  torch::jit::Code code(_constexpr_graph, "");
  torch::jit::InterpreterState state(code);

  state.run(*stack);

  ERROR_ON(_ins_to_make_consts.size() != stack->size());
}

void ConstExprEvaluator::replaceWithConstants(const torch::jit::Stack &stack) {
  // Cache the mapping of output value to stack output index
  std::map<torch::jit::Value *, size_t> constexpr_value_to_out_idx;
  for (size_t idx = 0; idx < _constexpr_graph->outputs().size(); idx++) {
    constexpr_value_to_out_idx[_constexpr_graph->outputs()[idx]] = idx;
  }

  for (auto value : _ins_to_make_consts) {
    // Find the matching stack output for the input from the constexpr
    auto constexpr_value = _values_map[value];

    // Obtain the resolved value from the stack
    auto resolved_value = stack.at(constexpr_value_to_out_idx[constexpr_value]);

    // Insert a constant to replace the original node and replace all uses
    torch::jit::WithInsertPoint insert_point(value->node());
    torch::jit::Value *new_const = _graph->insertConstant(resolved_value);
    value->replaceAllUsesWith(new_const);
  }
}

bool ConstExprEvaluator::nodeIsConstExpr(const torch::jit::Node &node) const {
  // If a node has no outputs, it may be a sentinel
  if (node.outputs().empty()) {
    return false;
  }

  // update_param_inplace has an output but will fail on node.hasSideEffects()
  if (node.kind() == symbols::poptorch::update_param_inplace) {
    return false;
  }

  // Random nodes or nodes with side effects cannot be constants
  if (isNondeterministic(node) || node.hasSideEffects()) {
    return false;
  }

  // Either the node has no inputs, or all inputs are outputs of nodes already
  // copied to the constexpres_graph
  for (auto input : node.inputs()) {
    if (_values_map.count(input) == 0) {
      return false;
    }
  }

  return true;
}

void ConstExprEvaluator::removeUnusedNodes() {
  // Iterate in reverse so that each node has no users
  for (auto node_it = _graph->nodes().rbegin();
       node_it != _graph->nodes().end(); node_it++) {
    if (_nodes_map.count(*node_it) != 0u) {
      node_it.destroyCurrent();
    }
  }
}

void ConstExprEvaluator::copyNodeToConstexprGraph(torch::jit::Node *node) {
  auto new_node = _constexpr_graph->createClone(
      node, [this](torch::jit::Value *v) { return this->_values_map[v]; },
      false);
  _nodes_map[node] = new_node;

  _constexpr_graph->insertNode(new_node);

  // Map the old outputs to the new
  auto old_it = node->outputs().begin();
  auto new_it = new_node->outputs().begin();
  for (; old_it != node->outputs().end(); old_it++, new_it++) {
    ERROR_ON(new_it == new_node->outputs().end());
    _values_map[*old_it] = *new_it;
  }
}

} // namespace

void evaluateConstexprs(torch::jit::Graph *graph) {
  ConstExprEvaluator evaluator(graph);
  evaluator.evaluate();
}

} // namespace type_and_constant_canonicalization
} // namespace poptorch
