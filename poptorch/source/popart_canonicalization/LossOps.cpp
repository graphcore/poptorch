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

  return createIdentityloss(graph, {square->output()}, reduction);
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
                       ignore_index);
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

  torch::jit::Node *log_x = createLog(graph, {x});

  // Log(x)*y
  torch::jit::Node *log_x_mul_y = createMul(graph, {y, log_x->output()});

  // Do (1 - y) and (1 - x)
  torch::jit::Node *x_minus_one = createSub(graph, {one->output(), x});
  torch::jit::Node *y_minus_one = createSub(graph, {one->output(), y});

  // Log(1 - x)
  torch::jit::Node *log_x_minus_one = createLog(graph, {x_minus_one->output()});

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

  final_node = createNeg(graph, {final_node->output()});

  return createIdentityloss(graph, {final_node->output()}, reduction);
}

torch::jit::Node *klDivHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // aten::kl_div(Tensor self, Tensor target, int reduction, bool log_target)

  // Input
  torch::jit::Value *x = node->input(0);
  // Target
  torch::jit::Value *y = node->input(1);
  std::int64_t reduction = constantToLong(node->input(2)->node());
  // Convert to popart reduce values
  reduction = convertReduceToPopart(reduction);
  // Whether the target is passed as log-probabilities
  bool log_target = constantToBool(node->input(3)->node());

  // log(y)
  torch::jit::Value *log_y;
  // Handle log-space targets at this stage
  if (log_target) {
    log_y = y;
    y = createExp(graph, {y})->output();
  } else {
    log_y = createLog(graph, {y})->output();
  }

  // log(y) - x
  torch::jit::Node *log_y_minus_x = createSub(graph, {log_y, x});

  // y(log(y) - x)
  torch::jit::Node *final_node = createMul(graph, {y, log_y_minus_x->output()});

  return createIdentityloss(graph, {final_node->output()}, reduction);
}

torch::jit::Node *poissonNllLossHandler(torch::jit::Graph *graph,
                                        torch::jit::Node *node) {
  // aten::poisson_nll_loss(Tensor input, Tensor target, bool log_input,
  //                        bool full, float eps, int reduction)

  // Input
  torch::jit::Value *x = node->input(0);
  // Target
  torch::jit::Value *y = node->input(1);
  // Whether the input is passed as log-probabilities
  bool log_input = constantToBool(node->input(2)->node());
  // Whether to compute full loss using Stirling approximation
  bool full = constantToBool(node->input(3)->node());
  // Added to avoid log(0) when log_input == false
  torch::jit::Value *epsilon = node->input(4);

  std::int64_t reduction = constantToLong(node->input(5)->node());
  // Convert to popart reduce values
  reduction = convertReduceToPopart(reduction);

  // log(x)
  torch::jit::Value *log_x;
  // Handle log-space inputs at this stage
  if (log_input) {
    log_x = x;
    x = createExp(graph, {x})->output();
  } else {
    torch::jit::Value *x_plus_eps = createAdd(graph, {x, epsilon})->output();
    log_x = createLog(graph, {x_plus_eps})->output();
  }

  // y log(x)
  torch::jit::Node *y_mul_log_x = createMul(graph, {y, log_x});

  // x - y log(x)
  torch::jit::Node *final_node = createSub(graph, {x, y_mul_log_x->output()});

  // Stirling approximation term = y log(y) − y + 0.5 log(2πy)
  if (full) {
    // log(y)
    torch::jit::Node *log_y = createLog(graph, {y});
    // y log(y)
    torch::jit::Node *y_mul_log_y = createMul(graph, {y, log_y->output()});
    // y log(y) - y
    torch::jit::Node *minus_y = createSub(graph, {y_mul_log_y->output(), y});

    // 2π
    torch::jit::Node *two_pi = createConstantFloat(graph, {2 * M_PI}, {});
    // 2πy
    torch::jit::Node *two_pi_y = createMul(graph, {two_pi->output(), y});
    // log(2πy)
    torch::jit::Node *log_two_pi_y = createLog(graph, {two_pi_y->output()});
    // 0.5
    torch::jit::Node *half = createConstantFloat(graph, {0.5}, {});
    // 0.5 log(2πy)
    torch::jit::Node *mul_half =
        createMul(graph, {half->output(), log_two_pi_y->output()});

    // y log(y) - y + 0.5 log(2πy)
    torch::jit::Node *add =
        createAdd(graph, {minus_y->output(), mul_half->output()});

    // Approximation values only added for target values > 1
    std::vector<std::int64_t> shape = shapeFromTensor(y);
    torch::jit::Node *ones = createConstantFloat(graph, {1}, shape);
    torch::jit::Node *mask = createGreater(graph, {y, ones->output()});
    torch::jit::Node *zeros = createConstantFloat(graph, {0}, shape);
    torch::jit::Node *masked_fill =
        createWhere(graph, {mask->output(), add->output(), zeros->output()});

    // x - y log(x) + y log(y) - y + 0.5 log(2πy)
    final_node =
        createAdd(graph, {final_node->output(), masked_fill->output()});
  }

  return createIdentityloss(graph, {final_node->output()}, reduction);
}

torch::jit::Node *hingeEmbeddingLossHandler(torch::jit::Graph *graph,
                                            torch::jit::Node *node) {
  // aten::hinge_embedding_loss(Tensor input, Tensor target, float margin,
  //                            int reduction)

  // Input
  torch::jit::Value *x = node->input(0);
  // Target labels containing 1 or -1
  torch::jit::Value *y = node->input(1);
  // Margin
  torch::jit::Value *delta = node->input(2);

  std::int64_t reduction = constantToLong(node->input(3)->node());
  // Convert to popart reduce values
  reduction = convertReduceToPopart(reduction);

  std::vector<std::int64_t> shape = shapeFromTensor(x);

  // Δ - x
  torch::jit::Node *delta_minus_x = createSub(graph, {delta, x});
  // 0
  torch::jit::Node *zeros = createConstantFloat(graph, {0}, shape);
  // max(0, Δ - x)
  torch::jit::Node *max_delta_minus_x =
      createMax(graph, {zeros->output(), delta_minus_x->output()});

  // 1
  torch::jit::Node *ones = createConstantInt(graph, {1}, shape);
  // -1
  torch::jit::Node *neg_ones = createConstantFloat(graph, {-1}, shape);
  // if y = 1
  torch::jit::Node *ones_mask = createEqual(graph, {y, ones->output()});
  // if y = -1
  torch::jit::Node *neg_ones_mask = createEqual(graph, {y, neg_ones->output()});

  // l = x              if y = 1
  torch::jit::Node *ones_masked_fill =
      createWhere(graph, {ones_mask->output(), x, zeros->output()});
  // l = max(0, Δ - x)  if y = -1
  torch::jit::Node *neg_ones_masked_fill =
      createWhere(graph, {neg_ones_mask->output(), max_delta_minus_x->output(),
                          zeros->output()});

  torch::jit::Node *final_node = createAdd(
      graph, {ones_masked_fill->output(), neg_ones_masked_fill->output()});

  return createIdentityloss(graph, {final_node->output()}, reduction);
}

torch::jit::Node *bceWithLogitsHandler(torch::jit::Graph *graph,
                                       torch::jit::Node *node) {
  // aten::binary_cross_entropy_with_logits(Tensor input, Tensor target,
  //                                        Tensor? weight, Tensor? pos_weight,
  //                                        int reduction)

  // Input
  torch::jit::Value *x = node->input(0);
  // Target
  torch::jit::Value *y = node->input(1);
  // Weight
  torch::jit::Value *w = node->input(2);
  // Weight of positive examples
  torch::jit::Value *pos_w = node->input(3);

  std::int64_t reduction = constantToLong(node->input(4)->node());
  // Convert to popart reduce values
  reduction = convertReduceToPopart(reduction);

  // -x
  torch::jit::Node *loss = createNeg(graph, {x});
  // 0
  torch::jit::Node *zeros = createConstantFloat(graph, {0}, {});
  // m = min(-x, 0)
  torch::jit::Node *m = createMin(graph, {loss->output(), zeros->output()});

  // -x - m
  loss = createSub(graph, {loss->output(), m->output()});
  // exp(-x - m)
  loss = createExp(graph, {loss->output()});

  // -m
  torch::jit::Node *neg_m = createNeg(graph, {m->output()});
  // exp(-m)
  torch::jit::Node *exp_neg_m = createExp(graph, {neg_m->output()});

  // exp(-m) + exp(-x - m)
  loss = createAdd(graph, {exp_neg_m->output(), loss->output()});
  // log(exp(-m) + exp(-x - m))
  loss = createLog(graph, {loss->output()});
  // m + log(exp(-m) + exp(-x - m))
  loss = createAdd(graph, {m->output(), loss->output()});

  // 1
  torch::jit::Node *ones = createConstantFloat(graph, {1}, {});

  // if pos_weight is specified
  if (!isNone(pos_w)) {
    // p - 1
    torch::jit::Node *p_minus_one = createSub(graph, {pos_w, ones->output()});
    // (p - 1) y
    torch::jit::Node *p_minus_one_mul_y =
        createMul(graph, {p_minus_one->output(), y});
    // l_p = (p - 1) y + 1
    torch::jit::Node *l_p =
        createAdd(graph, {p_minus_one_mul_y->output(), ones->output()});

    // l_p (m + log(exp(-m) + exp(-x - m)))
    loss = createMul(graph, {l_p->output(), loss->output()});
  }

  // (1 - y)
  torch::jit::Node *one_minus_y = createSub(graph, {ones->output(), y});
  // (1 - y) x
  torch::jit::Node *mul_x = createMul(graph, {one_minus_y->output(), x});
  // (1 - y) x + l_p (m + log(exp(-m) + exp(-x - m)))
  loss = createAdd(graph, {mul_x->output(), loss->output()});

  // if weight is specified
  if (!isNone(w)) {
    // w [(1 - y) x + l_p (m + log(exp(-m) + exp(-x - m)))]
    loss = createMul(graph, {w, loss->output()});
  }

  return createIdentityloss(graph, {loss->output()}, reduction);
}
} // namespace

// clang-format off
static bool handlers =
    registerHandlers(
        c10::aten::l1_loss, l1LossHandler,
        c10::aten::nll_loss, nllLossHandler,
        c10::aten::mse_loss, mseLossHandler,
        c10::aten::binary_cross_entropy, binaryCrossEntropyHandler,
        c10::aten::kl_div, klDivHandler,
        c10::aten::poisson_nll_loss, poissonNllLossHandler,
        c10::aten::hinge_embedding_loss, hingeEmbeddingLossHandler,
        c10::aten::binary_cross_entropy_with_logits, bceWithLogitsHandler,
        symbols::poptorch::identity_loss, identityLossHandler);
// clang-format on

} // namespace poptorch
