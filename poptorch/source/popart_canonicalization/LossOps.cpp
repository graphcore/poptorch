// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include <poptorch/OpBuilder.hpp>
#include <poptorch_logging/Error.hpp>
#include <poptorch_logging/Logging.hpp>

namespace poptorch {
namespace {
torch::jit::Node *l1LossHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  std::int64_t reduction =
      *handleConstant<std::int64_t>(node->input(2)->node());

  // Convert to popart reduce values.
  reduction = convertReduceToPopart(reduction);

  // Popart calculates the L1 loss as being the difference from an input to
  // 0. So we have to manually subract the losses first.
  torch::jit::Node *subtract =
      createSub(graph, {node->input(0), node->input(1)});

  const float scale = 1.0f;
  return createL1loss(graph, {subtract->output()}, scale, reduction);
}

torch::jit::Node *mseLossHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  std::int64_t reduction =
      *handleConstant<std::int64_t>(node->input(2)->node());

  // Convert to popart reduce values.
  reduction = convertReduceToPopart(reduction);

  // Subtract X - Y
  torch::jit::Node *subtract =
      createSub(graph, {node->input(0), node->input(1)});

  // Square it.
  torch::jit::Node *square =
      createMul(graph, {subtract->output(), subtract->output()});

  torch::jit::Node *final_node = square;

  if (reduction == 0) {
    // Sum
    final_node = createSum(graph, {square->output()});
  } else if (reduction == 1) {
    // Mean
    final_node = createMean(graph, {square->output()});
  }

  return createIdentityloss(graph, {final_node->output()}, reduction);
}

torch::jit::Node *nllLossHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  // This is derived by me (stephenm@graphcore.ai) not parsed from the
  // pytorch headers like the others as I can't find it in them.

  // "aten::nll_loss(Tensor input, Tensor label, Tensor? weight, int
  // reduction, int ignore_index) -> Tensor"

  std::int64_t reduction =
      *handleConstant<std::int64_t>(node->input(3)->node());
  std::int64_t ignore_index =
      *handleConstant<std::int64_t>(node->input(4)->node());

  // Convert to popart reduce values.
  reduction = convertReduceToPopart(reduction);

  return createNllloss(graph, {node->input(0), node->input(1)}, reduction,
                       ignore_index);
}
} // namespace

static bool handlers =
    registerHandlers(c10::aten::l1_loss, l1LossHandler, c10::aten::nll_loss,
                     nllLossHandler, c10::aten::mse_loss, mseLossHandler);

} // namespace poptorch
