// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {
torch::jit::Node *l1LossHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  std::int64_t reduction = constantToLong(node->input(2)->node());

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
  std::int64_t reduction = constantToLong(node->input(2)->node());

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

  std::int64_t reduction = constantToLong(node->input(3)->node());
  std::int64_t ignore_index = constantToLong(node->input(4)->node());

  // Convert to popart reduce values.
  reduction = convertReduceToPopart(reduction);

  return createNllloss(graph, {node->input(0), node->input(1)}, reduction,
                       ignore_index, /*inputIsLogProbability=*/true);
}

torch::jit::Node *identityLossHandler(torch::jit::Graph *graph,
                                      torch::jit::Node *node) {
  std::int64_t reduction = constantToLong(node->input(1)->node());

  return createIdentityloss(graph, {node->input(0)}, reduction);
}

torch::jit::Node *binaryCrossEntropyHandler(torch::jit::Graph *graph,
                                            torch::jit::Node *node) {
  // aten::binary_cross_entropy(Tensor input, Tensor target,
  //                            Tensor? weight, int reduction)

  // L = loss, w = weight, y= target, x = input.
  // Algorithm is: L = - w * (y *log(x) + (1 - y)*log(1 - x))

  // The input.
  torch::jit::Value *x = node->input(0);

  // The target.
  torch::jit::Value *y = node->input(1);

  // Optional weight term.
  torch::jit::Value *weight = node->input(2);

  // Loss reduction.
  std::int64_t reduction = constantToLong(node->input(3)->node());

  // Convert to popart reduce values.
  reduction = convertReduceToPopart(reduction);

  // Add the one constant
  torch::jit::Node *one = createConstantFloat(graph, {1.0}, {});

  torch::jit::Node *log_x = createUnarySameTypedOutput(createLog, graph, {x});

  // Log(x)*y
  torch::jit::Node *log_x_mul_y = createMul(graph, {y, log_x->output()});

  // Do (1 - y) and (1 - x)
  torch::jit::Node *x_minus_one = createSub(graph, {one->output(), x});
  torch::jit::Node *y_minus_one = createSub(graph, {one->output(), y});

  // Log(1 - x)
  torch::jit::Node *log_x_minus_one =
      createUnarySameTypedOutput(createLog, graph, {x_minus_one->output()});

  // (1 -y)*Log(1 - x)
  torch::jit::Node *subs_multiplied =
      createMul(graph, {y_minus_one->output(), log_x_minus_one->output()});

  // Log(x)*y + (1 -y)*Log(1 - x)
  torch::jit::Node *add_terms =
      createAdd(graph, {log_x_mul_y->output(), subs_multiplied->output()});

  torch::jit::Node *final_node = add_terms;

  if (weight->node()->kind() != c10::prim::Constant) {
    final_node = createMul(graph, {add_terms->output(), weight});
  }

  final_node =
      createUnarySameTypedOutput(createNeg, graph, {final_node->output()});
  if (reduction == 0) {
    // Sum
    final_node = createSum(graph, {final_node->output()});
  } else if (reduction == 1) {
    // Mean
    final_node = createMean(graph, {final_node->output()});
  }

  return createIdentityloss(graph, {final_node->output()}, reduction);
}
} // namespace

// clang-format off
static bool handlers =
    registerHandlers(
        c10::aten::l1_loss, l1LossHandler,
        c10::aten::nll_loss, nllLossHandler,
        c10::aten::mse_loss, mseLossHandler,
        c10::aten::binary_cross_entropy, binaryCrossEntropyHandler,
        symbols::poptorch::identity_loss, identityLossHandler);
// clang-format on

} // namespace poptorch
