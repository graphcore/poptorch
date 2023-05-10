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
  bool used_in_subgraph = false;
  WithNodeMetadata meta(node);

  // If this node is NOT used in the terminator then we need to add it as an
  // input to the graph.
  for (torch::jit::Value *value : node->inputs()) {
    // If the user isn't used in this subgraph AND the node hasn't already
    // been marked an input.
    if (subgraph->nodes.count(value->node()) == 0) {
      if (subgraph->added_inputs.count(value) == 0) {
        if (!inputFromParent) {
          torch::jit::Node *new_out = createAddUntypedInputTensor(graph, value);
          subgraph->input_map.insert({new_out->output(), value});
          subgraph->reverse_input_map.insert({value, new_out->output()});
        }
        subgraph->added_inputs.insert(value);
        used_in_subgraph = true;
      }
    } else {
      used_in_subgraph = true;
    }
  }

  return used_in_subgraph;
}

void markOutputs(torch::jit::Graph *graph, torch::jit::Node *outputs,
                 torch::jit::Node *insertion_point, Subgraph *subgraph) {
  torch::jit::WithInsertPoint insert_point(outputs);

  // Sometimes the return might not be processed in this node.
  const bool used_in_subgraph =
      markInputsAsComingFromParent(graph, outputs, subgraph);

  for (torch::jit::Value *output : outputs->inputs()) {
    WithNodeMetadata meta{output->node()};
    // Add an identity op in lieu if the op isn't used in the subgraph to make
    // sure popart handles the alias correctly.
    if (!used_in_subgraph) {
      torch::jit::Node *node = createIdentity(graph, {output});
      output = node->output();
    }

    // PopART doesn't allow inputs to be outputs directly.
    if (subgraph->reverse_input_map.find(output) !=
        subgraph->reverse_input_map.end()) {
      output = subgraph->reverse_input_map[output];
    }

    torch::jit::Node *new_node = createAddOutputTensor(graph, output);
    insertNodeBeforeNode(new_node, insertion_point);
  }
}

struct InsertionPointAndShape {
  torch::jit::Node *insertion_point;
  std::vector<std::int64_t> shape;
};
using ReshapePutterHelper = std::vector<InsertionPointAndShape>;

void markCondOutputs(torch::jit::Graph *graph, torch::jit::Node *outputs,
                     torch::jit::Node *insertion_point, Subgraph *subgraph,
                     ReshapePutterHelper &reshape_putter_helper,
                     bool processingElseOutputs = false) {
  torch::jit::WithInsertPoint insert_point(outputs);

  // Sometimes the return might not be processed in this node.
  const bool used_in_subgraph =
      markInputsAsComingFromParent(graph, outputs, subgraph);

  at::ArrayRef<torch::jit::Value *> inputs = outputs->inputs();
  for (size_t idx = 0; idx < inputs.size(); idx++) {
    torch::jit::Value *output = inputs[idx];

    WithNodeMetadata meta{output->node()};

    // Output tensor shape has to be read before adding IdentityOp as the shape
    // info does not propagate to the op output.
    const auto output_shape = shapeFromTensor(output);

    // Add an identity op in lieu if the op isn't used in the subgraph to make
    // sure popart handles the alias correctly.
    if (!used_in_subgraph) {
      torch::jit::Node *node = createIdentity(graph, {output});
      output = node->output();
    }

    // PopART doesn't allow inputs to be outputs directly.
    if (subgraph->reverse_input_map.find(output) !=
        subgraph->reverse_input_map.end()) {
      output = subgraph->reverse_input_map[output];
    }

    if (processingElseOutputs) {
      // Processing the else branch of the cond op. Here we make sure the
      // outputs of the branches have the same shapes. If not, we add a reshape
      // in the `then` branch.
      const auto &then_out_shape = reshape_putter_helper[idx].shape;
      const auto &else_out_shape = output_shape;
      if (else_out_shape.empty()) {
        ERROR("`else` branch output has an empty shape, so adding a reshape "
              "op to the `then` branch to achieve shapes identity is not "
              "possible!");
      }

      if (then_out_shape != else_out_shape) {
        // In case if branches output shapes differ, there is a reshape added:
        // 1. Create a reshape op
        torch::jit::Node *reshape_node = nullptr;
        {
          torch::jit::WithInsertPoint reshape_insert_point(
              reshape_putter_helper[idx].insertion_point);
          auto *tensor_to_reshape =
              reshape_putter_helper[idx].insertion_point->input();
          reshape_node =
              createReshape(graph, tensor_to_reshape, else_out_shape);
        }

        // 2. Create a new output tensor of the `then` branch (being the reshape
        // output) and insert it before the original output tensor op.
        torch::jit::Node *new_then_output_node =
            createAddOutputTensor(graph, reshape_node->output());
        insertNodeBeforeNode(new_then_output_node,
                             reshape_putter_helper[idx].insertion_point);

        // 3. Remove the original output tensor op returning the wrongly shaped
        // tensor.
        reshape_putter_helper[idx].insertion_point->destroy();
      }
      // Create the output tensor of the `else` branch.
      torch::jit::Node *else_output_node = createAddOutputTensor(graph, output);
      insertNodeBeforeNode(else_output_node, insertion_point);

    } else {
      // Create the output tensor of the `then` branch.
      // In case the output tensor turns out to be of a different shape then
      // `else` branch'es one, it will be replaced with the reshaped output
      // tensor.
      torch::jit::Node *then_output_node = createAddOutputTensor(graph, output);
      insertNodeBeforeNode(then_output_node, insertion_point);

      reshape_putter_helper.push_back(
          {then_output_node, shapeFromTensor(output)});
    }
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
void annotateSubgraphs(torch::jit::Graph *graph, torch::jit::Node *start_node) {
  logging::LogContext ctx_func("annotateSubgraphs Processing");
  // Subgraph start to all nodes contained directly within that subgraph.
  std::stack<Subgraph> subgraph_nodes;

  // Nodes to delete (if they are truely unused).
  std::unordered_set<torch::jit::Node *> to_delete;

  // Helper struct for processing if_else.
  std::stack<ReshapePutterHelper> reshape_putter_helpers_stack;

  // Look for any subgraphs. Subgraphs are currently:
  // * for loops.
  for (auto iter = start_node->iterator(); iter != graph->nodes().end();
       ++iter) {
    torch::jit::Node *node = *iter;
    logging::LogContext ctx("Processing " + nodeToString(node));
    const torch::jit::Symbol kind = node->kind();

    if (kind == symbols::poptorch::start_for_loop) {
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
    } else if (kind == symbols::poptorch::start_if_block) {
      // Start tracking the new subgraph.
      subgraph_nodes.push(Subgraph());

      // Delete the input node (condition) as it is not needed anymore.
      to_delete.insert(node->input(0)->node());
      node->removeInput(0);
    } else if (kind == symbols::poptorch::start_else_block) {
      // Add the outputs of `then` branch just before starting the else block.
      reshape_putter_helpers_stack.emplace();
      markCondOutputs(graph, node->input(0)->node(), node,
                      &subgraph_nodes.top(), reshape_putter_helpers_stack.top(),
                      false /*processingElseOuputs*/);

      // Remove the if subgraph.
      subgraph_nodes.pop();

      // Start tracking the new subgraph.
      subgraph_nodes.push(Subgraph());

      // Delete the input node (then_branch output), as it is not needed
      // anymore.
      to_delete.insert(node->input(0)->node());
      node->removeInput(0);
    } else if (kind == symbols::poptorch::end_if_block) {
      // Mark the outputs of the `else` block.
      markCondOutputs(graph, node->input(0)->node(), node,
                      &subgraph_nodes.top(), reshape_putter_helpers_stack.top(),
                      true /*processingElseOutputs*/);
      reshape_putter_helpers_stack.pop();

      // Remove the else subgraph.
      subgraph_nodes.pop();

      // Record the number of outputs.
      const std::size_t num_outputs = node->input(0)->node()->inputs().size();
      node->i_(c10::Symbol::fromQualString("attr::num_outputs"), num_outputs);

      // Delete the 1st input node (else_branch output), as it is not needed
      // anymore.
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

} // namespace poptorch
