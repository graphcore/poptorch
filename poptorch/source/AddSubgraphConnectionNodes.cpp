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
  // All the nodes in the subgraph.
  std::unordered_set<torch::jit::Node *> nodes;

  // Track the inputs already added so we don't double count them.
  std::unordered_set<torch::jit::Value *> added_inputs;

  // Map of new inputs to old inputs.
  std::unordered_map<torch::jit::Value *, torch::jit::Value *> input_map;

  // Map of old inputs to the new ones.
  std::unordered_map<torch::jit::Value *, torch::jit::Value *>
      reverse_input_map;
};

bool isTerminator(const torch::jit::Node *node) {
  return node->kind() == symbols::poptorch::end_for_loop;
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
      if (!inputFromParent) {
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
  torch::jit::WithInsertPoint insert_point(outputs);

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

// State during the dispatcher intercept calls.
std::stack<torch::jit::Node *> start_for_loop_nodes;

} // namespace

/*
 * Certain ops are essentially subgraphs within the main graph. For instance
 * for loops. If they have a tensor which comes from the subgraph
 * above we must add a specific input entry op to the graph for that op.
 */
void annotateSubgraphs(torch::jit::Graph *graph, torch::jit::Node *start_node,
                       bool training) {
  logging::LogContext ctx_func("annotateSubgraphs Processing");
  // Subgraph start to all nodes contained directly within that subgraph.
  std::stack<Subgraph> subgraph_nodes;

  // Nodes to delete (if they are truely unused).
  std::unordered_set<torch::jit::Node *> to_delete;

  // Look for any subgraphs. Subgraphs are currently:
  // * for loops.
  for (auto iter = start_node->iterator(); iter != graph->nodes().end();
       ++iter) {
    torch::jit::Node *node = *iter;
    logging::LogContext ctx("Processing " + nodeToString(node));
    const torch::jit::Symbol kind = node->kind();

    if (kind == symbols::poptorch::start_for_loop) {
      ERROR_ON_MSG(training,
                   "poptorch.for_loop() is only supported in inference.");
      // Start tracking the new subgraph.
      subgraph_nodes.push(Subgraph());

      torch::jit::WithInsertPoint insert_point(node->next());
      markInputsAsComingFromParent(graph, node->input()->node(),
                                   &subgraph_nodes.top(), false);

      // We no longer need these inputs.
      to_delete.insert(node->input(0)->node());
      node->removeInput(0);

    } else if (kind == symbols::poptorch::end_for_loop) {
      markOutputs(graph, node->input(0)->node(), node, &subgraph_nodes.top());
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
      torch::jit::WithInsertPoint insert_point(node);
      markInputsAsComingFromParent(graph, node, &subgraph_nodes.top());

      for (const std::pair<torch::jit::Value *const, torch::jit::Value *>
               &pair : subgraph_nodes.top().input_map) {
        node->replaceInputWith(pair.second, pair.first);
      }
    }
  }

  for (torch::jit::Node *node : to_delete) {
    if (node->output()->uses().empty()) {
      node->destroy();
    }
  }
}

// Same pass as annotateSubgraphs above but working as a state machine, using
// start_for_loop_nodes as state between the calls.
void annotateSubgraphsDispatch(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  logging::LogContext ctx_func("annotateSubgraphsDispatch");

  const torch::jit::Symbol kind = node->kind();

  if (kind == symbols::poptorch::start_for_loop) {
    start_for_loop_nodes.push(node);
  } else if (kind == symbols::poptorch::end_for_loop) {
    ERROR_ON_MSG(start_for_loop_nodes.empty(),
                 "Internal: end_for_loop encountered before end_for_loop");

    auto *start_node = start_for_loop_nodes.top();
    // Whenever loop body has inplace ops that change the input of the body,
    // the graph we see is incorrect. It is incorrect because during loop
    // lowering we use the inputs of poptorch::end_for_loop which comes after
    // the inplace ops in the graph and hence has the input-changing op's output
    // as an input. We fix this by simply overwriting the input of
    // poptorch::end_for_loop with the input of poptorch::start_for_loop which
    // points to the correct ssa value because it comes before the inplace ops
    // in the graph.
    if (node->input(1) != start_node->input()) {
      node->replaceInput(1, start_node->input());
    }

    if (start_for_loop_nodes.size() == 1) {
      // TODO(T51159): Add support for dispatch tracing + training.
      // Currently we just pass training = false to annotateSubgraphs.
      annotateSubgraphs(graph, start_node, /*training=*/false);
    }

    start_for_loop_nodes.pop();
  }
}

} // namespace poptorch
