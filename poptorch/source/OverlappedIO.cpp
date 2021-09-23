// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include <algorithm>

#include "PoptorchSymbols.hpp"
#include "poptorch/OverlappedIO.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {

void attributiseOverlappedIO(torch::jit::Graph *graph) {
  std::set<torch::jit::Node *> to_erase_output_and_delete;
  std::vector<torch::jit::Node *> to_delete;

  int64_t input_num = -1;
  for (auto input : graph->inputs()) {
    input_num++;
    auto input_uses = input->uses();

    // Sort by topoligical
    std::sort(input_uses.begin(), input_uses.end(),
              [](const torch::jit::Use &first, const torch::jit::Use &second) {
                return first.user->isBefore(second.user);
              });

    if (input_uses.empty()) {
      continue;
    }

    auto &first_user(input_uses[0].user);

    if ((input_uses.size() == 1) &&
        (first_user->kind() ==
         poptorch::symbols::poptorch::set_overlap_for_input)) {
      auto value_node = first_user->input(1)->node();
      ERROR_ON(value_node->kind() != c10::prim::Constant);
      const auto &value_str = value_node->s(c10::attr::value);
      to_delete.push_back(first_user);
      first_user->removeInput(1);

      // String constant may be shared
      to_erase_output_and_delete.insert(value_node);

      graph->param_node()->s_(getOverlapSymbol(input_num), value_str);
      first_user->output()->replaceAllUsesWith(input);
      continue;
    }

    // This should be the only op
    for (const auto &other_use : input_uses) {
      ERROR_ON_MSG(
          other_use.user->kind() ==
              poptorch::symbols::poptorch::set_overlap_for_input,
          "poptorch.set_overlap_for_input must be the only op applied to an "
          "input. This is not the case for input "
              << input->debugName() << " to the model.");
    }
  }

  for (auto node : to_erase_output_and_delete) {
    node->eraseOutput(0);
    node->destroy();
  }

  for (torch::jit::Node *node : to_delete) {
    node->destroy();
  }

  // Any other use of set_overlap_for_input is invalid
  for (auto node : graph->nodes()) {
    ERROR_ON_MSG(node->kind() ==
                     poptorch::symbols::poptorch::set_overlap_for_input,
                 "poptorch.set_overlap_for_input applied on a node which is "
                 "not a tensor "
                 "input to the model.");
  }
}

} // namespace poptorch
