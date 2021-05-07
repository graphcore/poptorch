// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>
#include <vector>

#include "poptorch/AliasProcessing.hpp"

namespace poptorch {

void resolveAliases(torch::jit::Graph *graph) {
  std::vector<torch::jit::Node *> to_delete;

  for (auto node : graph->nodes()) {
    if (node->kind() != c10::aten::alias) {
      continue;
    }

    // Replace the alias output with the direct input
    node->output()->replaceAllUsesWith(node->input());
    to_delete.push_back(node);
  }

  for (auto node : to_delete) {
    node->destroy();
  }
}
} // namespace poptorch
