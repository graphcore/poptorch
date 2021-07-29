// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include "poptorch/EliminateListConstructs.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

bool isAppendNode(torch::jit::Node *node) {
  return node->kind() == c10::aten::append;
}

bool isConstantNode(torch::jit::Node *node) {
  return node->kind() == c10::prim::Constant;
}

template <typename T> T getValue(torch::jit::Node *node);

template <> int64_t getValue(torch::jit::Node *node) {
  auto sym = c10::attr::value;
  return node->i(sym);
}

template <typename T>
torch::jit::Node *
createConstantFromList(const std::vector<torch::jit::Node *> &appendNodes) {
  std::vector<T> values;
  for (auto node : appendNodes) {
    T value = getValue<T>(node->input(1)->node());
    values.push_back(value);
  }

  auto x = appendNodes.at(0);
  auto graph = x->owningGraph();
  torch::jit::Value *const_node_output =
      graph->insertConstant(values, x->sourceRange(), x->scope());
  return const_node_output->node();
}

bool tryCreateConstantNode(torch::jit::Node *node) {
  auto output = node->output();

  // Collect all uses of the output
  std::unordered_set<torch::jit::Node *> unordered_uses;
  for (auto use : output->uses()) {
    unordered_uses.insert(use.user);
  }

  // If the list construct is never used, do nothing.
  if (unordered_uses.empty()) {
    return false;
  }

  // Check that all uses are in the same block.
  for (auto use : unordered_uses) {
    if (use->owningBlock() != node->owningBlock()) {
      logging::err("Not attempting to handle case where not all uses are in "
                   "the same block.");
      return false;
    }
  }

  // Get the uses in the order they appear in the block.
  std::vector<torch::jit::Node *> uses;
  for (auto n : node->owningBlock()->nodes()) {
    if (unordered_uses.find(n) != unordered_uses.end()) {
      uses.push_back(n);
    }
  }

  // There should be an initial block of appends at the start of uses, and then
  // no further appends after that.
  int index_of_last_initial_append = -1;
  for (uint i = 0; i < uses.size(); i++) {
    auto use = uses[i];
    if (isAppendNode(use)) {
      index_of_last_initial_append = i;
    } else {
      break;
    }
  }
  for (uint i = index_of_last_initial_append + 1; i < uses.size(); i++) {
    auto use = uses[i];
    if (isAppendNode(use)) {
      logging::err(
          "There is an append after list has already been used by another "
          "node. This list can not be replaced by a constant.");
      return false;
    }
  }

  // Collect all the append nodes
  std::vector<torch::jit::Node *> append_nodes;
  for (auto use : uses) {
    if (isAppendNode(use)) {
      append_nodes.push_back(use);
    }
  }

  // All append nodes should take an output of a constant node as an input.
  for (auto n : append_nodes) {
    auto x = n->input(1);
    if (!isConstantNode(x->node())) {
      logging::err(
          "Input to append node is not output of a prim::Constant node.");
      return false;
    }
  }

  auto type = output->type();
  torch::jit::Node *constant_node;
  if (type->isSubtypeOf(torch::jit::ListType::ofInts())) {
    constant_node = createConstantFromList<int64_t>(append_nodes);
  } else {
    logging::err("Unhandled output type {}", *type);
    return false;
  }
  constant_node->moveAfter(node);
  output->replaceAllUsesAfterNodeWith(append_nodes.back(),
                                      constant_node->output());
  return true;
}

void tryDeleteListConstructNode(torch::jit::Node *listConstruct) {
  auto error_message = [&](const std::string &msg) {
    logging::err("Unable to remove prim::ListConstruct node: {}  {}",
                 *listConstruct, msg);
  };

  // Collect all the nodes that append values to the ListConstruct.
  std::vector<torch::jit::Node *> append_nodes;
  for (auto use : listConstruct->output()->uses()) {
    auto x = use.user;
    if (!isAppendNode(x)) {
      std::stringstream ss;
      ss << "Node is still used by something other than an aten::append node("
         << *x << ")";
      error_message(ss.str());
      return;
    }

    if (!x->output()->uses().empty()) {
      std::stringstream ss;
      ss << "Output of append node(" << *x << ") still has a use.";
      error_message(ss.str());
      return;
    }

    append_nodes.push_back(x);
  }

  // Collect all the prim::Constant nodes that are inputs into the append nodes.
  std::vector<torch::jit::Node *> constant_inputs;
  for (auto a : append_nodes) {
    auto x = a->input(1)->node();
    if (!isConstantNode(x)) {
      error_message("Input to aten::append node is not a prim::Constant.");
      return;
    }

    if (x->output()->uses().size() > 1) {
      error_message("Output of prim::Constant is used by more than the "
                    "aten::append node.");
      return;
    }

    constant_inputs.push_back(x);
  }

  for (auto x : append_nodes) {
    x->destroy();
  }
  for (auto x : constant_inputs) {
    x->destroy();
  }
  listConstruct->destroy();
}

void eliminateListConstructs(torch::jit::Block *block) {
  std::vector<torch::jit::Node *> to_delete;

  for (auto node : block->nodes()) {
    logging::LogContext ctx("eliminateListConstructs Processing " +
                            nodeToString(node));
    if (node->kind() == c10::prim::ListConstruct) {
      if (tryCreateConstantNode(node)) {
        to_delete.push_back(node);
      }
    }
  }

  for (auto node : to_delete) {
    tryDeleteListConstructNode(node);
  }
}

void eliminateListConstructs(torch::jit::Graph *graph) {
  eliminateListConstructs(graph->block());
}

} // namespace poptorch
