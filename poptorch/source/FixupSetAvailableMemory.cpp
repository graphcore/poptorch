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

} // namespace

void setAvailableMemoryAddPossibleInputOp(torch::jit::Node *node) {
  ERROR_ON(!isValidInputOpForAMP(node));
  logging::trace("Adding node {} as a possible input to set_available_memory",
                 nodeToString(node));
  amp_possible_input_nodes.push_back(node);
}

void setAvailableMemoryFixupInput(torch::jit::Node *set_available_memory_node) {
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
  // inputs of the input. We don't want to do a full search in the graph as it
  // might lead to undesired results. The search will be limited to the possible
  // grandparent nodes that are made of decomposed ops such as scatteradd and
  // linear. These ops are composed of multiple ops, and set_available_memory
  // needs to find the suitable op among them.
  auto parent_inputs = current_input_node->inputs();
  const auto *last_parent_input = parent_inputs.end();
  const auto *grandparent_iter =
      std::find_if(parent_inputs.begin(), last_parent_input, [](Value *input) {
        return std::any_of(
            amp_possible_input_nodes.rbegin(), amp_possible_input_nodes.rend(),
            [grandparent = input->node()](Node *possible_grandparent) {
              return possible_grandparent == grandparent;
            });
      });
  if (grandparent_iter == last_parent_input) {
    logging::trace(
        "No matching grandparent found for set_available_memory node {}",
        nodeToString(set_available_memory_node));
    return;
  }

  Value *new_input = *grandparent_iter;
  auto *current_input = set_available_memory_node->input(0);
  logging::trace("Replacing set_available_memory input '%{}' with '%{}'",
                 current_input->debugName(), new_input->debugName());
  set_available_memory_node->replaceInputWith(
      set_available_memory_node->input(), new_input);
  tryRemovePossibleInput(new_input->node());
}

void setAvailableMemoryOnGraphFinalized() {
  logging::trace("Clearing list of possible inputs ops to AMP");
  amp_possible_input_nodes.clear();
}

} // namespace poptorch
