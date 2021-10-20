// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *addHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  // aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor
  // aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) ->
  // (Tensor(a!))
  torch::jit::Value *alpha_param = node->input(2);

  // If both types are bool, use logical_or
  if (allInputsBool(node, 2)) {
    ERROR_ON(!hasUnityValue(alpha_param));
    return createLogical_or(graph, {node->input(0), node->input(1)});
  }

  // Ordinary addition
  torch::jit::Value *alpha_multiplicand = node->input(1);
  if (!hasUnityValue(alpha_param)) {
    auto *alpha_node = createMul(graph, {alpha_param, alpha_multiplicand});
    alpha_multiplicand = alpha_node->output();
  }
  return createAdd(graph, {node->input(0), alpha_multiplicand});
}

torch::jit::Node *truncHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // Drop the exponent by casting to int and back.
  torch::jit::Node *to_int = createCast(graph, node->input(), c10::kInt);

  return createCast(
      graph, to_int->output(),
      *node->input()->type()->expect<c10::TensorType>()->scalarType());
}

torch::jit::Node *fracHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // Frac(x) = x - trunc(x)

  // Drop the exponent by casting to int and back.
  torch::jit::Node *to_int = createCast(graph, node->input(), c10::kInt);

  torch::jit::Node *trunc = createCast(
      graph, to_int->output(),
      *node->input()->type()->expect<c10::TensorType>()->scalarType());

  return createSub(graph, {node->input(), trunc->output()});
}

torch::jit::Node *floorDivideHandler(torch::jit::Graph *graph,
                                     torch::jit::Node *node) {
  // aten::floor_divide(Tensor x, Tensor y) -> Tensor
  // floor_divide(x, y) = floor(x/y) where floor(...) rounds towards 0

  torch::jit::Node *quotient =
      createDiv(graph, {node->input(0), node->input(1)});

  torch::jit::Node *cast = createCast(graph, quotient->output(), c10::kInt);

  return createCast(
      graph, cast->output(),
      *quotient->output()->type()->expect<c10::TensorType>()->scalarType());
}

torch::jit::Node *mulHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  // aten::mul(Tensor self, Tensor other) -> Tensor

  // If both types are bool, use logical_add
  if (allInputsBool(node)) {
    return createLogical_and(graph, {node->input(0), node->input(1)});
  }

  // Ordinary multiplication
  return createMul(graph, {node->input(0), node->input(1)});
}

torch::jit::Node *trueDivideHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // aten::true_divide(Tensor x, Tensor y) -> Tensor
  // true_divide(x, y) = (float)x / (float)y

  torch::jit::Node *x = createCast(graph, node->input(0), c10::kFloat);

  torch::jit::Node *y = createCast(graph, node->input(1), c10::kFloat);

  return createDiv(graph, {x->output(), y->output()});
}

torch::jit::Node *clampHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto *x = node->input(0);
  auto *min_node = node->input(1)->node();
  auto min = isNone(min_node) ? std::numeric_limits<float>::lowest()
                              : constantToFloat(min_node);

  auto *max_node = node->input(2)->node();
  auto max = isNone(max_node) ? std::numeric_limits<float>::max()
                              : constantToFloat(max_node);
  return createClip(graph, {x}, max, min);
}

torch::jit::Node *clampMinHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  auto *max = graph->createNone()->output();
  auto *input = node->input(0);
  auto *min = node->input(1);
  auto clamp_handler = getHandler(c10::aten::clamp);
  return createHandlerOperation(graph, clamp_handler, {input, min, max});
}

torch::jit::Node *clampMaxHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  auto *min = graph->createNone()->output();
  auto *input = node->input(0);
  auto *max = node->input(1);
  auto clamp_handler = getHandler(c10::aten::clamp);
  return createHandlerOperation(graph, clamp_handler, {input, min, max});
}

torch::jit::Node *addCDivHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  torch::jit::Node *div = createDiv(graph, {node->input(1), node->input(2)});
  auto scale = constantToFloat(node->input(3)->node());
  torch::jit::Node *scaled = createScale(graph, {div->output()}, scale);
  return createAdd(graph, {node->input(0), scaled->output()});
}

torch::jit::Node *crossHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  auto *x = node->input(0);
  auto *y = node->input(1);
  auto *opt_axis = node->input(2)->node();

  auto x_shape = shapeFromTensor(x);
  auto y_shape = shapeFromTensor(y);

  ERROR_ON_MSG(x_shape.size() != y_shape.size(),
               "Cross product tensors must have same rank");
  for (unsigned i = 0; i < x_shape.size(); ++i) {
    ERROR_ON_MSG(x_shape[i] != y_shape[i],
                 "Cross product tensors must have same shape");
  }

  unsigned axis = 0;
  if (isNone(opt_axis)) {
    // if unspecified, the axis is the first to have dimension 3
    for (unsigned i = 0; i < x_shape.size(); ++i) {
      if (x_shape[i] == 3) {
        axis = i;
        break;
      }
    }
  } else {
    axis = constantToInt(opt_axis);
  }
  ERROR_ON_MSG(x_shape[axis] != 3,
               "Cross product product axis must have dimension 3");

  auto *indices = createConstantInt(graph, {2, 0, 1}, {3})->output();

  // circular permutation right by 1 along the axis
  auto *x_roll = createGather(graph, {x, indices}, axis)->output();
  auto *y_roll = createGather(graph, {y, indices}, axis)->output();

  // products of one straight input with the other input permuted
  auto *mul_x_y_roll = createMul(graph, {x, y_roll})->output();
  auto *mul_y_x_roll = createMul(graph, {y, x_roll})->output();

  // subtraction produces result permuted one position left
  auto *result_roll = createSub(graph, {mul_y_x_roll, mul_x_y_roll})->output();

  // permute to compute final result
  return createGather(graph, {result_roll, indices}, axis);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::add, addHandler);
  registerHandler(c10::aten::add_, addHandler);
  registerHandler(c10::aten::trunc, truncHandler);
  registerHandler(c10::aten::frac, fracHandler);
  registerHandler(c10::aten::floor_divide, floorDivideHandler);
  registerHandler(c10::aten::mul, mulHandler);
  registerHandler(c10::aten::true_divide, trueDivideHandler);
  registerHandler(c10::aten::clamp, clampHandler);
  registerHandler(c10::aten::clamp_min, clampMinHandler);
  registerHandler(c10::aten::clamp_max, clampMaxHandler);
  registerHandler(c10::aten::addcdiv, addCDivHandler);
  registerHandler(c10::aten::cross, crossHandler);
}

} // namespace poptorch
