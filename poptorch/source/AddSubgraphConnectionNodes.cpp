// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include <functional>
#include <stack>

#include "PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

namespace {

// A small class to keep track of information regarding subgraphs.
struct Subgraph {
  Subgraph() : nodes({}), added_inputs({}), input_map({}), is_loop(false) {}

  explicit Subgraph(bool loop)
      : nodes({}), added_inputs({}), input_map({}), is_loop(loop) {}

  // All the nodes in the subgraph.
  std::unordered_set<torch::jit::Node *> nodes;

  // Track the inputs already added so we don't double count them.
  std::unordered_set<torch::jit::Value *> added_inputs;

  // Map of new inputs to old inputs.
  std::unordered_map<torch::jit::Value *, torch::jit::Value *> input_map;

  // Map of old inputs to the new ones.
  std::unordered_map<torch::jit::Value *, torch::jit::Value *>
      reverse_input_map;

  bool is_loop;
};

bool isTerminator(const torch::jit::Node *node) {
  return node->kind() == symbols::poptorch::end_if ||
         node->kind() == symbols::poptorch::end_for_loop;
}

bool isUsedInTerminator(const torch::jit::Node *node) {
  for (const torch::jit::Value *output : node->outputs()) {
    for (const torch::jit::Use &use : output->uses()) {
      const torch::jit::Node *user = use.user;

      if (isTerminator(user)) {
        return true;
      }
    }
  }

  return false;
}

bool markInputsAsComingFromParent(torch::jit::Graph *graph,
                                  torch::jit::Node *node, Subgraph *subgraph,
                                  const bool inputFromParent = true) {
  bool changed = false;

  // If this node is NOT used in the terminator then we need to add it as an
  // input to the graph.
  for (torch::jit::Value *value : node->inputs()) {
    // If the user isn't used in this subgraph AND the node hasn't already
    // been marked an input.
    if (subgraph->nodes.count(value->node()) == 0 &&
        subgraph->added_inputs.count(value) == 0) {
      if (inputFromParent) {
        // For some reason popart doens't support loops having this attribute.
        if (!subgraph->is_loop) {
          createAddInputTensorFromParentGraph(graph, value);
        }
      } else {
        torch::jit::Node *new_out = createAddUntypedInputTensor(graph, value);
        subgraph->input_map.insert({new_out->output(), value});
        subgraph->reverse_input_map.insert({value, new_out->output()});
      }
      subgraph->added_inputs.insert(value);
      changed = true;
    }
  }

  return changed;
}

void markOutputs(torch::jit::Graph *graph, torch::jit::Node *outputs,
                 torch::jit::Node *insertion_point, Subgraph *subgraph) {
  graph->setInsertPoint(outputs);

  // Sometimes the return might not be processed in this node.
  bool not_used_in_subgraph =
      markInputsAsComingFromParent(graph, outputs, subgraph);

  for (torch::jit::Value *output : outputs->inputs()) {
    // Add an identity op in lieu if the op isn't used in the subgraph to make
    // sure popart handles the alias correctly.
    if (not_used_in_subgraph) {
      torch::jit::Node *node = createIdentity(graph, {output});
      output = node->output();
    }

    // PopART doesn't allow inputs to be outputs directly.
    if (subgraph->reverse_input_map.find(output) !=
        subgraph->reverse_input_map.end()) {
      output = subgraph->reverse_input_map[output];
    }

    torch::jit::Node *new_node = createAddOutputTensor(graph, output);

    new_node->insertBefore(insertion_point);
  }
}

} // namespace
/*
 * Certain ops are essentially subgraphs within the main graph. For instance
 * if/else and while loops. If they have tensor which comes from the subgraph
 * above we must add a specific input entry op to the graph for that op.
 */
void annotateSubgraphs(torch::jit::Graph *graph) {
  logging::LogContext ctx_func("annotateSubgraphs Processing");
  // Subgraph start to all nodes contained directly within that subgraph.
  std::stack<Subgraph> subgraph_nodes;

  // Nodes to delete (if they are truely unused).
  std::unordered_set<torch::jit::Node *> to_delete;

  // Look for any subgraphs. Subgraphs are currently:
  // * If/Else bodies.
  for (torch::jit::Node *node : graph->nodes()) {
    logging::LogContext ctx("Processing " + nodeToString(node));
    const torch::jit::Symbol kind = node->kind();

    // Both if and else will create a new subgraph.
    if (kind == symbols::poptorch::start_if_true ||
        kind == symbols::poptorch::start_if_false) {
      // We propagate the outputs to the else branch so they can be handled
      // before dealing with the new branch.
      if (kind == symbols::poptorch::start_if_false) {
        // Add the outputs of the if just before this.
        markOutputs(graph, node->input(0)->node(), node, &subgraph_nodes.top());

        // Delete the node.
        to_delete.insert(node->input(0)->node());
        node->removeInput(0);
      }

      // Start tracking the new subgraph.
      subgraph_nodes.push({});

    } else if (kind == symbols::poptorch::end_if) {
      // Mark the outputs of the else.
      markOutputs(graph, node->input(1)->node(), node, &subgraph_nodes.top());

      // Remove the else.
      subgraph_nodes.pop();

      // Remove the if.
      subgraph_nodes.pop();

      const std::size_t num_outputs = node->input(1)->node()->inputs().size();

      // Once these are remove this will be the last use (since we construct the
      // list).
      to_delete.insert(node->input(1)->node());

      // We no longer need these inputs.
      node->removeInput(1);

      // Record the number of outputs.
      node->i_(c10::Symbol::fromQualString("attr::num_outputs"), num_outputs);
    } else if (kind == symbols::poptorch::start_for_loop) {
      // Start tracking the new subgraph.
      subgraph_nodes.push(Subgraph(true));

      graph->setInsertPoint(node->next());
      markInputsAsComingFromParent(graph, node->input()->node(),
                                   &subgraph_nodes.top(), false);

      // We no longer need these inputs.
      to_delete.insert(node->input(0)->node());
      node->removeInput(0);

    } else if (kind == symbols::poptorch::end_for_loop) {
      // Mark the outputs of the else.
      markOutputs(graph, node->input(0)->node(), node, &subgraph_nodes.top());

      // Remove the else.
      subgraph_nodes.pop();

      // We no longer need these inputs.
      to_delete.insert(node->input(0)->node());
      node->removeInput(0);
    } else if (kind == symbols::poptorch::add_untyped_input_tensor) {
      continue;
    } else if (!subgraph_nodes.empty()) {
      // Don't count the list construction nodes.
      if (isUsedInTerminator(node)) {
        continue;
      }

      // Add this node to the active subgraph.
      subgraph_nodes.top().nodes.insert(node);
      graph->setInsertPoint(node);
      markInputsAsComingFromParent(graph, node, &subgraph_nodes.top());

      if (!isUsedInTerminator(node)) {
        for (const std::pair<torch::jit::Value *const, torch::jit::Value *>
                 &pair : subgraph_nodes.top().input_map) {
          node->replaceInputWith(pair.second, pair.first);
        }
      }
    }
  }

  for (torch::jit::Node *node : to_delete) {
    if (node->output()->uses().empty()) {
      node->destroy();
    }
  }
}

} // namespace poptorch
