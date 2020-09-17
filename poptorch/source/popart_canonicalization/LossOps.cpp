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

torch::jit::Node *nllLoss2dHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  // "aten::nll_loss2d(Tensor input, Tensor target, Tensor height, Tensor
  // weight, int reduction, int ignore_index) -> Tensor"

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

  std::int64_t reduction = constantToLong(node->input(3)->node());
  std::int64_t ignore_index = constantToLong(node->input(4)->node());

  reduction = convertReduceToPopart(reduction);

  torch::jit::Value *in = node->input(0); // in for input
  torch::jit::Value *target = node->input(1);

  std::vector<std::int64_t> shape_input = shapeFromTensor(in);
  std::vector<std::int64_t> shape_target = shapeFromTensor(target);

  std::int64_t n = shape_input[0];
  std::int64_t c = shape_input[1];
  std::int64_t height = shape_input[2];
  std::int64_t width = shape_input[3];
  std::int64_t n_1 = shape_target[0];
  std::int64_t height_1 = shape_target[1];
  std::int64_t width_1 = shape_target[2];
  std::int64_t flat = n * height * width;
  ERROR_ON_MSG(
      n != n_1,
      "Dimension size mismatch: the parameter N from input[0] and target[0]");
  ERROR_ON_MSG(height != height_1,
               "Dimension size mismatch: the input height and target height");
  ERROR_ON_MSG(width != width_1,
               "Dimension size mismatch: the input width and target width");

  std::vector<std::int64_t> input_new_shape({flat, c});
  std::vector<std::int64_t> target_new_shape({flat});

  torch::jit::Node *perm = createTranspose(graph, {in}, {0, 2, 3, 1});
  torch::jit::Node *reshape_input =
      createReshape(graph, perm->output(), input_new_shape);

  torch::jit::Node *reshape_target =
      createReshape(graph, target, target_new_shape);

  torch::jit::Node *final_node =
      createNllloss(graph, {reshape_input->output(), reshape_target->output()},
                    reduction, ignore_index, /*inputIsLogProbability=*/true);

  if (reduction == 2) {
    // If "none" reduction, return the results with input's original shape
    return createReshape(graph, final_node->output(), shape_target);
  }

  return final_node;
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
        c10::aten::nll_loss2d, nllLoss2dHandler,
        c10::aten::mse_loss, mseLossHandler,
        c10::aten::binary_cross_entropy, binaryCrossEntropyHandler,
        symbols::poptorch::identity_loss, identityLossHandler);
// clang-format on

} // namespace poptorch
