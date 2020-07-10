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

void canonicalizeLists(torch::jit::Graph *graph) {
  std::vector<torch::jit::Node *> to_delete;

  // 1st pass. Looking for broadcasts.
  for (torch::jit::Node *node : graph->nodes()) {
    const torch::jit::Symbol kind = node->kind();
    const std::string kind_as_str = kind.toDisplayString();

    if (kind_as_str == "aten::broadcast_tensors") {
      node->output()->replaceAllUsesWith(node->input());
      to_delete.push_back(node);
    }
  }

  // Delete the nodes we need to delete.
  for (torch::jit::Node *node : to_delete) {
    node->destroy();
  }
  to_delete.clear();

  // 2nd pass. Hitting the actual loops.
  for (torch::jit::Node *list : graph->nodes()) {
    const torch::jit::Symbol kind = list->kind();
    const std::string kind_as_str = kind.toDisplayString();

    // Eliminate lists with just an "unpack" as their user.
    if (kind_as_str == "prim::ListConstruct") {
      if (list->output()->uses().size() == 1) {
        torch::jit::Node *unpack = list->output()->uses()[0].user;

        const std::string unpack_as_str = unpack->kind().toDisplayString();
        // Make sure it is an unpack.
        if (unpack_as_str == "prim::ListUnpack") {
          for (std::uint32_t i = 0; i < unpack->outputs().size(); ++i) {
            // Replace each output of the unpack with the input of the original
            // list.
            unpack->outputs()[i]->replaceAllUsesWith(list->inputs()[i]);
          }

          to_delete.push_back(unpack);
          to_delete.push_back(list);
        }
      }
    }
  }

  // Delete the nodes we need to delete.
  for (torch::jit::Node *node : to_delete) {
    node->destroy();
  }
}

} // namespace poptorch
