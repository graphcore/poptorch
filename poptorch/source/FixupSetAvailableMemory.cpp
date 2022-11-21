// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/graph_node_list.h>
#include <torch/csrc/jit/ir/ir.h>

#include <algorithm>
#include <vector>

#include "PoptorchSymbols.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

using torch::jit::Node;
using torch::jit::Value;

namespace poptorch {

namespace {
std::vector<Node *> amp_possible_input_nodes;

bool isValidInputOpForAMP(const Node *node) {
  namespace popart_syms = poptorch::symbols::popart;
  const auto kind = node->kind();
  return kind == popart_syms::gather || kind == popart_syms::lstm ||
         kind == popart_syms::matmul || kind == popart_syms::scatter ||
         kind == popart_syms::scatterreduce;
}

// Returns true if the given node was removed by searching the possible inputs
// backwards.
bool tryRemovePossibleInput(const Node *input) {
  auto input_nodes_count = amp_possible_input_nodes.size();
  auto remove_position = std::remove(amp_possible_input_nodes.rbegin(),
                                     amp_possible_input_nodes.rend(), input);
  amp_possible_input_nodes.erase(amp_possible_input_nodes.rend().base(),
                                 remove_position.base());
  return input_nodes_count > amp_possible_input_nodes.size();
}

torch::jit::Value *tryFindAncestor(torch::jit::Value *v, int depth_to_check,
                                   int depth = 0) {
  if (depth == depth_to_check) {
    if (tryRemovePossibleInput(v->node())) {
      return v;
    }
  }
  for (auto *inp : v->node()->inputs()) {
    if (torch::jit::Value *ancestor =
            tryFindAncestor(inp, depth_to_check, depth + 1)) {
      return ancestor;
    }
  }
  return nullptr;
}

} // namespace

void setAvailableMemoryAddPossibleInputOp(torch::jit::Node *node) {
  if (!isValidInputOpForAMP(node)) {
    return;
  }
  logging::trace("Adding node {} as a possible input to set_available_memory",
                 nodeToString(node));
  amp_possible_input_nodes.push_back(node);
}

void moveSetAvailableMemoryIfRequired(
    torch::jit::Node *set_available_memory_node) {
  ERROR_ON(set_available_memory_node->kind() !=
           poptorch::symbols::poptorch::set_available_memory);
  if (amp_possible_input_nodes.empty()) {
    return;
  }

  // If the current input is already in the possible inputs list, remove it,
  // and return.
  Node *current_input_node = set_available_memory_node->input(0)->node();
  if (tryRemovePossibleInput(current_input_node)) {
    return;
  }
  logging::trace("Found set_available_memory node that might need fixup: {}",
                 nodeToString(set_available_memory_node));

  // The current input isn't among the possible inputs. Try to go through the
  // inputs of the input.
  //
  // If we don't find anything: try one more level. (In some cases there is
  // a reshape followed by an add).
  //
  // We don't want to do a full search in the graph as it
  // might lead to undesired results. The search will be limited to the
  // possible grandparent and great grandparent nodes that are made of
  // decomposed ops such as scatteradd and linear. These ops are composed
  // of multiple ops, and set_available_memory needs to find the suitable
  // op among them.
  torch::jit::Value *new_input =
      tryFindAncestor(set_available_memory_node->input(0), 1);
  if (new_input == nullptr) {
    new_input = tryFindAncestor(set_available_memory_node->input(0), 2);
  }
  if (new_input == nullptr) {
    logging::trace(
        "No matching ancestor found for set_available_memory node {}",
        nodeToString(set_available_memory_node));
    return;
  }

  auto *current_input = set_available_memory_node->input(0);
  logging::trace("Replacing set_available_memory input '%{}' with '%{}'",
                 current_input->debugName(), new_input->debugName());
  // Remove set_available_memory_node from its current position
  set_available_memory_node->output()->replaceAllUsesWith(
      set_available_memory_node->input(0));

  // Replace all the uses of the new input with "set_available_memory"
  new_input->replaceAllUsesWith(set_available_memory_node->output());
  // Update set_available_memory's input
  set_available_memory_node->moveAfter(new_input->node());
  set_available_memory_node->replaceInput(0, new_input);
}

} // namespace poptorch
