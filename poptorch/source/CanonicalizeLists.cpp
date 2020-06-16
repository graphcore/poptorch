// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

/*
 * No lint due to the linter expecting PopartLateCanonicalization.hpp which
 * rightly doesn't exist.
 */
#include <functional> // NOLINT

#include <poptorch/OpBuilder.hpp>              // NOLINT
#include <poptorch/PopartCanonicalization.hpp> // NOLINT
#include <torch/csrc/jit/ir/ir.h>              // NOLINT

namespace poptorch {

void CanonicalizeLists(torch::jit::Graph &graph) {
  std::vector<torch::jit::Node *> toDelete;

  // 1st pass. Looking for broadcasts.
  for (torch::jit::Node *node : graph.nodes()) {
    const torch::jit::Symbol kind = node->kind();
    const std::string kindAsStr = kind.toDisplayString();

    if (kindAsStr == "aten::broadcast_tensors") {
      node->output()->replaceAllUsesWith(node->input());
      toDelete.push_back(node);
    }
  }

  // Delete the nodes we need to delete.
  for (torch::jit::Node *node : toDelete) {
    node->destroy();
  }
  toDelete.clear();

  // 2nd pass. Hitting the actual loops.
  for (torch::jit::Node *list : graph.nodes()) {
    const torch::jit::Symbol kind = list->kind();
    const std::string kindAsStr = kind.toDisplayString();

    // Eliminate lists with just an "unpack" as their user.
    if (kindAsStr == "prim::ListConstruct") {
      if (list->output()->uses().size() == 1) {
        torch::jit::Node *unpack = list->output()->uses()[0].user;

        const std::string unpackAsStr = unpack->kind().toDisplayString();
        // Make sure it is an unpack.
        if (unpackAsStr == "prim::ListUnpack") {
          for (std::int32_t i = 0; i < unpack->outputs().size(); ++i) {
            // Replace each output of the unpack with the input of the original
            // list.
            unpack->outputs()[i]->replaceAllUsesWith(list->inputs()[i]);
          }

          toDelete.push_back(unpack);
          toDelete.push_back(list);
        }
      }
    }
  }

  // Delete the nodes we need to delete.
  for (torch::jit::Node *node : toDelete) {
    node->destroy();
  }
}

} // namespace poptorch
