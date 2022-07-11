// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include <algorithm>

#include "PoptorchSymbols.hpp"
#include "poptorch/OverlappedIO.hpp"
#include "poptorch/Utils.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

namespace {
void attributiseOverlappedInputs(
    torch::jit::Graph *graph,
    std::set<torch::jit::Node *> *to_erase_output_and_delete,
    std::vector<torch::jit::Node *> *to_delete, bool verify_only) {
  logging::LogContext ctx("attributiseOverlappedInputs");

  int64_t input_num = -1;
  for (auto *input : graph->inputs()) {
    input_num++;
    auto input_uses = input->uses();

    if (input_uses.empty()) {
      continue;
    }

    auto &user(input_uses[0].user);

    if ((input_uses.size() == 1) &&
        (user->kind() == poptorch::symbols::poptorch::set_overlap_for_input)) {
      auto *value_node = user->input(1)->node();
      if (!verify_only) {
        ERROR_ON(value_node->kind() != c10::prim::Constant);
        const auto &value_str = value_node->s(c10::attr::value);
        graph->param_node()->s_(getOverlapSymbol("input", input_num),
                                value_str);
      }
      to_delete->push_back(user);
      user->removeInput(1);

      // String constant may be shared
      if (value_node->output()->uses().empty()) {
        to_erase_output_and_delete->insert(value_node);
      }

      user->output()->replaceAllUsesWith(input);
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
}

void errorOnDoubleReturnOfOutput(torch::jit::Node *node) {
  logging::LogContext ctx("check double return of" + nodeToString(node));
  uint32_t return_count = 0;

  std::function<void(torch::jit::Value *)> count_returns;
  count_returns = [&count_returns,
                   &return_count](torch::jit::Value *input_value) {
    for (auto use : input_value->uses()) {
      if (use.user->kind() ==
              poptorch::symbols::poptorch::set_overlap_for_output ||
          use.user->kind() == c10::prim::ListConstruct ||
          use.user->kind() == c10::prim::TupleConstruct) {
        count_returns(use.user->output());
      } else if (use.user->kind() == c10::prim::Return) {
        return_count++;
      }
    }
  };

  count_returns(node->input(0));

  ERROR_ON(return_count == 0);

  ERROR_ON_MSG(
      return_count > 1,
      "poptorch.set_overlap_for_output cannot be "
      "used with a tensor that is returned twice. Please check all returned "
      "tensors including those nested in tuples/lists.");
}

void attributiseOverlappedOutputs(
    torch::jit::Graph *graph,
    std::set<torch::jit::Node *> *to_erase_output_and_delete,
    std::vector<torch::jit::Node *> *to_delete, bool verify_only) {
  logging::LogContext ctx("attributiseOverlappedOutputs");

  int64_t output_num = 0;

  std::function<void(torch::jit::Node *)> process_node;
  process_node = [&process_node, graph, &output_num, to_erase_output_and_delete,
                  to_delete, verify_only](torch::jit::Node *node) {
    auto overlap_symbol = getOverlapSymbol("output", output_num);
    if (node->kind() == poptorch::symbols::poptorch::set_overlap_for_output) {
      errorOnDoubleReturnOfOutput(node);

      auto *value_node = node->input(1)->node();
      if (!verify_only) {
        ERROR_ON(value_node->kind() != c10::prim::Constant);
        const auto &value_str = value_node->s(c10::attr::value);
        graph->return_node()->s_(overlap_symbol, value_str);
      }
      to_delete->push_back(node);
      node->removeInput(1);

      // String constant may be shared
      if (value_node->output()->uses().empty()) {
        to_erase_output_and_delete->insert(value_node);
      }

      node->output()->replaceAllUsesWith(node->input(0));
      output_num++;
    } else if (node->kind() == c10::prim::ListConstruct ||
               node->kind() == c10::prim::TupleConstruct) {
      for (auto *input : node->inputs()) {
        process_node(input->node());
      }
    } else {
      const std::string value_str = "no_overlap";
      graph->return_node()->s_(overlap_symbol, value_str);
      output_num++;
    }
  };

  // Loop over all graph (there may always only be one as multiple inputs are
  // returned as a tuple/list)
  for (auto *output : graph->outputs()) {
    process_node(output->node());
  }
}
} // namespace

void attributiseOverlappedIO(torch::jit::Graph *graph, bool verify_only) {
  std::set<torch::jit::Node *> to_erase_output_and_delete;
  std::vector<torch::jit::Node *> to_delete;

  attributiseOverlappedInputs(graph, &to_erase_output_and_delete, &to_delete,
                              verify_only);
  attributiseOverlappedOutputs(graph, &to_erase_output_and_delete, &to_delete,
                               verify_only);

  for (auto *node : to_erase_output_and_delete) {
    node->eraseOutput(0);
    node->destroy();
  }

  for (torch::jit::Node *node : to_delete) {
    node->destroy();
  }

  // Any other use of set_overlap_for_input or set_overlap_for_input is invalid
  for (auto *node : graph->nodes()) {
    ERROR_ON_MSG(node->kind() ==
                     poptorch::symbols::poptorch::set_overlap_for_input,
                 "poptorch.set_overlap_for_input applied on a node which is "
                 "not a tensor input to the model.");

    ERROR_ON_MSG(node->kind() ==
                     poptorch::symbols::poptorch::set_overlap_for_output,
                 "poptorch.set_overlap_for_output applied on a node which is "
                 "not a tensor output to the model.");
  }
}

void verifyOverlappedIOForDispatch(torch::jit::Graph *graph) {
  attributiseOverlappedIO(graph, /*verify_only=*/true);
}

} // namespace poptorch
