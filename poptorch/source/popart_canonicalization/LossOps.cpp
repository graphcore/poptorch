// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/DispatchTracer.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {
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
  torch::jit::Node *one = createConstantFloatLike(graph, x, {1.0}, {});

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

torch::jit::Node *nllLossNdHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // "aten::nll_loss2d(Tensor input, Tensor target, Tensor height, Tensor
  // weight, int reduction, int ignore_index) -> Tensor"

  // aten::nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int
  // reduction, int ignore_index) -> (Tensor output, Tensor total_weight)

  // aten::nll_loss2d() is implemented based on popart:nllloss().
  // Suppose the input[0] has the shape of (N, C, M, K)
  // input[0] will be transposed with perm [0, 2, 3, 1],
  //   and reshaped with (N * M * K, C), pushing C to the last dimension.
  // input[1] will be reshaped to (N * M * K), before calling nllloss.
  // The generated IRs are as follows:
  // %37 : Tensor = popart::transpose[perm=[0, 2, 3, 1]](%35)
  // %38 : Tensor(500:4, 4:1) = popart::reshape_static_shape[shape=[500,4]](%37)
  // %39 : Int(500:1) = popart::reshape_static_shape[shape=[500]](%25)
  // %40 : Float() = popart::nllloss[reduction=1, ignoreIndex=-100](%38, %39)

  // aten::nll_loss_nd(Tensor input, Tensor target, Tensor? weight, int
  // reduction, int ignore_index) -> Tensor
  std::int64_t reduction = constantToLong(node->input(3)->node());
  std::int64_t ignore_index = constantToLong(node->input(4)->node());

  reduction = convertReduceToPopart(reduction);

  torch::jit::Value *input = node->input(0);
  torch::jit::Value *target = node->input(1);
  torch::jit::Value *weight = node->input(2);
  // TODO(T42695): Support optional weight parameter
  ERROR_ON_MSG(!isNone(weight),
               "Parameter \"weight\" is unsupported for aten::nll_loss_nd");
  std::vector<std::int64_t> shape_input = shapeFromTensor(input);
  std::vector<std::int64_t> shape_target = shapeFromTensor(target);

  if (shape_input.size() != 2) {
    // Input shape: (N, C, d1, d2, ..., dk)
    // Target shape: (N, d1, d2, ..., dk)

    // Suppose the input has the shape of (N, C, M, K)
    // The input will be transposed with perm [0, 2, 3, 1],
    //   and reshaped with (N * M * K, C), pushing C to the last dimension.
    // The target will be reshaped to (N * M * K), before calling nllloss

    std::int64_t c = shape_input[1];
    std::int64_t flat =
        std::accumulate(shape_target.begin(), shape_target.end(), 1,
                        std::multiplies<int64_t>{});

    // Create an input permutation of (0, 2, 3, ..., N, 1)
    std::vector<int64_t> p(shape_input.size(), 0);
    std::iota(p.begin() + 1, p.end() - 1, 2);
    p[p.size() - 1] = 1;

    // Permute the class dimension to the end
    torch::jit::Node *perm = createTranspose(graph, {input}, p);

    input = createReshape(graph, perm->output(), {flat, c})->output();
    target = createReshape(graph, target, {flat})->output();
  }

  torch::jit::Node *loss =
      createNllloss(graph, {input, target}, reduction, ignore_index,
                    /*inputIsLogProbability=*/true);

  if (reduction == 2) {
    // If "none" reduction, return the results with target's original shape
    loss = createReshape(graph, loss->output(), shape_target);
  }

  return createIdentityloss(graph, {loss->output()}, reduction);
}

torch::jit::Node *crossEntropyLossHandler(torch::jit::Graph *graph,
                                          torch::jit::Node *node) {
  // aten::cross_entropy_loss(Tensor self, Tensor target, Tensor? weight, int
  // reduction, int ignore_index)
  auto *input = node->input(0);
  auto *target = node->input(1);
  auto *weight = node->input(2);
  // TODO(T42695): Support optional weight parameter
  ERROR_ON_MSG(
      !isNone(weight),
      "Parameter \"weight\" is unsupported for aten::cross_entropy_loss");
  auto *reduction = node->input(3);
  auto *ignore_index = node->input(4);

  auto log_softmax_handler = getHandler(c10::aten::log_softmax);
  auto *log_softmax = createHandlerOperation(
      graph, log_softmax_handler, {input, wrapInConstant1D(graph, 1)});
  // logSoftmaxHandler loses shape information required by nllLossNdHandler,
  // so we need to set the type to that of the input, as the type will be the
  // same
  log_softmax->output()->setType(input->type());

  return createHandlerOperation(
      graph, nllLossNdHandler,
      {log_softmax->output(), target, weight, reduction, ignore_index});
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
  torch::jit::Node *y_log_y_minus_x =
      createMul(graph, {y, log_y_minus_x->output()});

  // Handle any log(y) where y<=0 from earlier
  torch::jit::Node *zeros = createConstantFloatLike(graph, y, {0}, {});
  torch::jit::Node *mask = createGreater(graph, {y, zeros->output()});
  torch::jit::Node *final_node = createWhere(
      graph, {mask->output(), y_log_y_minus_x->output(), zeros->output()});

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

  // Stirling approximation term = y log(y) -y + 0.5 log(2*PI*y)
  if (full) {
    // log(y)
    torch::jit::Node *log_y = createLog(graph, {y});
    // y log(y)
    torch::jit::Node *y_mul_log_y = createMul(graph, {y, log_y->output()});
    // y log(y) - y
    torch::jit::Node *minus_y = createSub(graph, {y_mul_log_y->output(), y});

    // 2*PI
    torch::jit::Node *two_pi =
        createConstantFloatLike(graph, x, {2 * M_PI}, {});
    // 2*PI*y
    torch::jit::Node *two_pi_y = createMul(graph, {two_pi->output(), y});
    // log(2*PI*y)
    torch::jit::Node *log_two_pi_y = createLog(graph, {two_pi_y->output()});
    // 0.5
    torch::jit::Node *half = createConstantFloatLike(graph, x, {0.5}, {});
    // 0.5 log(2*PI*y)
    torch::jit::Node *mul_half =
        createMul(graph, {half->output(), log_two_pi_y->output()});

    // y log(y) - y + 0.5 log(2*PI*y)
    torch::jit::Node *add =
        createAdd(graph, {minus_y->output(), mul_half->output()});

    // Approximation values only added for target values > 1
    std::vector<std::int64_t> shape = shapeFromTensor(y);
    torch::jit::Node *ones = createConstantFloatLike(graph, x, {1}, shape);
    torch::jit::Node *mask = createGreater(graph, {y, ones->output()});
    torch::jit::Node *zeros = createConstantFloatLike(graph, x, {0}, shape);
    torch::jit::Node *masked_fill =
        createWhere(graph, {mask->output(), add->output(), zeros->output()});

    // x - y log(x) + y log(y) - y + 0.5 log(2*PI*y)
    final_node =
        createAdd(graph, {final_node->output(), masked_fill->output()});
  }

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
  torch::jit::Node *zeros = createConstantFloatLike(graph, x, {0}, {});

  // m = max(-x, 0)
  torch::jit::Node *m = createMax(graph, {loss->output(), zeros->output()});

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
  torch::jit::Node *ones = createConstantFloatLike(graph, x, {1}, {});

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

// TODO(T30688): Unsupported since the PyTorch implementation doesn't
//               currently use this aten function
torch::jit::Node *multiLabelSoftMarginLossHandler(torch::jit::Graph *graph,
                                                  torch::jit::Node *node) {
  // aten::multilabel_soft_margin_loss(Tensor input, Tensor target,
  //                                   Tensor? weight, int reduction)

  // Input
  torch::jit::Value *x = node->input(0);
  // Target
  torch::jit::Value *y = node->input(1);
  // Weight
  torch::jit::Value *w = node->input(2);

  std::int64_t reduction = constantToLong(node->input(3)->node());
  // Convert to popart reduce values
  reduction = convertReduceToPopart(reduction);

  auto log_sigmoid_handler = getHandler(c10::aten::log_sigmoid);

  // -x
  torch::jit::Node *loss = createNeg(graph, {x});
  // log(sigmoid(-x))
  loss = createHandlerOperation(graph, log_sigmoid_handler, {loss->output()});

  // 1
  torch::jit::Node *ones = createConstantFloatLike(graph, x, {1}, {});
  // 1 - y
  torch::jit::Node *one_minus_y = createSub(graph, {ones->output(), y});

  // (1 - y) log(sigmoid(-x))
  loss = createMul(graph, {one_minus_y->output(), loss->output()});

  // log(sigmoid(x))
  torch::jit::Node *log_sig_x =
      createHandlerOperation(graph, log_sigmoid_handler, {x});
  // y log(sigmoid(x))
  torch::jit::Node *y_mul_log_sig_x =
      createMul(graph, {y, log_sig_x->output()});

  // y log(sigmoid(x)) + (1 - y) log(sigmoid(-x))
  loss = createAdd(graph, {y_mul_log_sig_x->output(), loss->output()});
  // -(y log(sigmoid(x)) + (1 - y) log(sigmoid(-x)))
  loss = createNeg(graph, {loss->output()});

  // if weight is specified
  if (!isNone(w)) {
    // -w (y log(sigmoid(x)) + (1 - y) log(sigmoid(-x)))
    loss = createMul(graph, {w, loss->output()});
  }

  return createIdentityloss(graph, {loss->output()}, reduction);
}

torch::jit::Node *cosineEmbeddingLossHandler(torch::jit::Graph *graph,
                                             torch::jit::Node *node) {
  // aten::cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target,
  //                             float margin, int reduction)

  // Input 1
  torch::jit::Value *x1 = node->input(0);
  // Input 2
  torch::jit::Value *x2 = node->input(1);
  // Target
  torch::jit::Value *y = node->input(2);
  // Margin
  torch::jit::Value *margin = node->input(3);

  std::int64_t reduction = constantToLong(node->input(4)->node());
  // Convert to popart reduce values
  reduction = convertReduceToPopart(reduction);

  // Epsilon
  torch::jit::Value *epsilon =
      createConstantFloatLike(graph, x1, {1e-12}, {})->output();

  // x1 * x2
  torch::jit::Node *x1_mul_x2 = createMul(graph, {x1, x2});
  // sum(x1 * x2)
  torch::jit::Node *sum_x1_mul_x2 =
      createReducesum(graph, {x1_mul_x2->output()}, {1}, 0);

  // sum_sqr(x1)
  torch::jit::Node *sum_sqr_x1 = createReducesumsquare(graph, {x1}, {1}, 0);
  // sq1 = sum_sqr(x1) + eps
  torch::jit::Node *sum_sqr_x1_plus_eps =
      createAdd(graph, {sum_sqr_x1->output(), epsilon});

  // sum_sqr(x2)
  torch::jit::Node *sum_sqr_x2 = createReducesumsquare(graph, {x2}, {1}, 0);
  // sq2 = sum_sqr(x2) + eps
  torch::jit::Node *sum_sqr_x2_plus_eps =
      createAdd(graph, {sum_sqr_x2->output(), epsilon});

  // sq1 * sq1
  torch::jit::Node *sq1_mul_sq2 = createMul(
      graph, {sum_sqr_x1_plus_eps->output(), sum_sqr_x2_plus_eps->output()});
  // sqrt(sq1 * sq2)
  torch::jit::Node *sqrt_sq1_mul_sq2 =
      createSqrt(graph, {sq1_mul_sq2->output()});

  // cos_sim(x1, x2)
  torch::jit::Node *cos_sim =
      createDiv(graph, {sum_x1_mul_x2->output(), sqrt_sq1_mul_sq2->output()});

  // 1
  torch::jit::Node *ones = createConstantFloatLike(graph, x1, {1}, {});
  // 1 - cos_sim(x1, x2)
  torch::jit::Node *one_minus_cos_sim =
      createSub(graph, {ones->output(), cos_sim->output()});

  // cos_sim(x1, x2) - margin
  torch::jit::Node *cos_sim_minus_margin =
      createSub(graph, {cos_sim->output(), margin});
  // 0
  torch::jit::Node *zeros = createConstantFloatLike(graph, x1, {0}, {});
  // max(0, cos_sim(x1, x2) - margin)
  torch::jit::Node *max_zero_cos_sim_minus_margin =
      createMax(graph, {zeros->output(), cos_sim_minus_margin->output()});

  // -1
  torch::jit::Node *neg_ones = createConstantInt(graph, {-1}, {});
  // if y = 1
  torch::jit::Node *ones_mask = createEqual(graph, {y, ones->output()});
  // if y = -1
  torch::jit::Node *neg_ones_mask = createEqual(graph, {y, neg_ones->output()});

  // l = 1 - cos(x1, x2)               if y = 1
  torch::jit::Node *ones_masked_fill =
      createWhere(graph, {ones_mask->output(), one_minus_cos_sim->output(),
                          zeros->output()});
  // l = max(0, cos(x1, x2) - margin)  if y = -1
  torch::jit::Node *neg_ones_masked_fill = createWhere(
      graph, {neg_ones_mask->output(), max_zero_cos_sim_minus_margin->output(),
              zeros->output()});

  torch::jit::Node *loss = createAdd(
      graph, {ones_masked_fill->output(), neg_ones_masked_fill->output()});

  return createIdentityloss(graph, {loss->output()}, reduction);
}

torch::jit::Node *tripletMarginLossHandler(torch::jit::Graph *graph,
                                           torch::jit::Node *node) {
  // aten::triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative,
  //                           float margin, float p, float eps, bool swap, int
  //                           reduction)

  // Anchor
  torch::jit::Value *a = node->input(0);
  // Positive
  torch::jit::Value *pos = node->input(1);
  // Negative
  torch::jit::Value *neg = node->input(2);
  // Margin
  torch::jit::Value *margin = node->input(3);
  // Norm degree for pairwise distance
  torch::jit::Value *p = node->input(4);
  // Small value to avoid division by zero
  torch::jit::Value *eps = node->input(5);
  // Swap
  bool swap = constantToBool(node->input(6)->node());

  // keepdim = false
  torch::jit::Value *keepdim = createConstantInt(graph, {0}, {})->output();

  std::int64_t reduction = constantToLong(node->input(7)->node());
  // Convert to popart reduce values
  reduction = convertReduceToPopart(reduction);

  // pairwiseDistanceHandler
  auto pairwise_dist_handler = getHandler(c10::aten::pairwise_distance);

  // d(a, pos)
  torch::jit::Node *loss = createHandlerOperation(graph, pairwise_dist_handler,
                                                  {a, pos, p, eps, keepdim});
  // d(a, neg)
  torch::jit::Node *dist_neg = createHandlerOperation(
      graph, pairwise_dist_handler, {a, neg, p, eps, keepdim});

  if (swap) {
    torch::jit::Node *dist_swap = createHandlerOperation(
        graph, pairwise_dist_handler, {pos, neg, p, eps, keepdim});
    // d(a, neg) = min(d(a, neg), d(pos, neg))
    dist_neg = createMin(graph, {dist_neg->output(), dist_swap->output()});
  }

  // d(a, pos) - d(a, neg)
  loss = createSub(graph, {loss->output(), dist_neg->output()});
  // d(a, pos) - d(a, neg) + margin
  loss = createAdd(graph, {loss->output(), margin});

  torch::jit::Node *zeros = createConstantFloatLike(graph, a, {0}, {});
  // max(d(a, pos) - d(a, neg) + margin, 0)
  loss = createMax(graph, {loss->output(), zeros->output()});

  return createIdentityloss(graph, {loss->output()}, reduction);
}

torch::jit::Node *ctcLossHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  auto *log_probs = node->input(0);
  auto *targets = node->input(1);
  auto *input_lengths = node->input(2);
  auto *target_lengths = node->input(3);
  auto blank = constantToInt(node->input(4)->node());
  auto reduction = constantToLong(node->input(5)->node());
  auto zero_inf = constantToBool(node->input(6)->node());

  ERROR_ON_MSG(reduction == 0,
               "CTCLoss with reduction=\"none\" is currently not supported");

  targets = createCast(graph, {targets}, "UINT32")->output();
  if (input_lengths->type()->kind() == c10::TypeKind::TensorType) {
    // aten::ctc_loss.Tensor
    input_lengths = createCast(graph, {input_lengths}, "UINT32")->output();
  } else {
    // aten::ctc_loss.IntList: convert to tensor for popart::_ctcloss
    const auto values = constantToLongVec(input_lengths->node());
    const std::int64_t shape = values.size();
    input_lengths = createConstantInt(graph, values, {shape})->output();
  }

  if (target_lengths->type()->kind() == c10::TypeKind::TensorType) {
    // aten::ctc_loss.Tensor
    target_lengths = createCast(graph, {target_lengths}, "UINT32")->output();
  } else {
    // aten::ctc_loss.IntList: convert to tensor for popart::_ctcloss
    const auto values = constantToLongVec(target_lengths->node());
    const std::int64_t shape = values.size();
    target_lengths = createConstantInt(graph, values, {shape})->output();
  }

  reduction = convertReduceToPopart(reduction);
  auto *loss =
      create_ctcloss(graph, {log_probs, targets, input_lengths, target_lengths},
                     reduction, blank, "UNDEFINED", zero_inf);

  return createIdentityloss(graph, {loss->output()}, reduction);
}

torch::jit::Node *ctcbeamsearchdecoderHandler(torch::jit::Graph *graph,
                                              torch::jit::Node *node) {
  auto *log_probs = node->input(0);
  auto *lengths = node->input(1);
  auto blank = constantToInt(node->input(2)->node());
  auto width = constantToInt(node->input(3)->node());
  auto top_paths = constantToInt(node->input(4)->node());

  lengths = createCast(graph, {lengths}, "UINT32")->output();
  auto *decoder = createCtcbeamsearchdecoder(graph, {log_probs, lengths}, blank,
                                             width, top_paths);
  decoder->addOutput();
  decoder->addOutput();

  node->output(0)->replaceAllUsesWith(decoder->output(0));
  node->output(1)->replaceAllUsesWith(decoder->output(1));
  node->output(2)->replaceAllUsesWith(decoder->output(2));

  markNodeForDeletion(node);
  return decoder;
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::nll_loss2d, nllLossNdHandler);
  registerHandler(c10::aten::nll_loss2d_forward, nllLossNdHandler);
  registerHandler(c10::aten::nll_loss_nd, nllLossNdHandler);
  registerHandler(c10::aten::nll_loss_forward, nllLossNdHandler);
  registerHandler(c10::aten::binary_cross_entropy, binaryCrossEntropyHandler);
  registerHandler(c10::aten::kl_div, klDivHandler);
  registerHandler(c10::aten::poisson_nll_loss, poissonNllLossHandler);
  registerHandler(c10::aten::binary_cross_entropy_with_logits,
                  bceWithLogitsHandler);
  registerHandler(c10::aten::multilabel_soft_margin_loss,
                  multiLabelSoftMarginLossHandler);
  registerHandler(c10::aten::cosine_embedding_loss, cosineEmbeddingLossHandler);
  registerHandler(c10::aten::triplet_margin_loss, tripletMarginLossHandler);
  registerHandler(c10::aten::ctc_loss, ctcLossHandler);
  registerHandler(symbols::poptorch::ctc_beam_search_decoder,
                  ctcbeamsearchdecoderHandler);
  registerHandler(c10::aten::cross_entropy_loss, crossEntropyLossHandler);
}

} // namespace poptorch
