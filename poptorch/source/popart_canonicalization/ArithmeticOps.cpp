// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *addHandler(torch::jit::Graph *graph, torch::jit::Node *node) {
  // aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor

  torch::jit::Value *alpha_param = node->input(2);

  // If both types are bool, use logical_or
  if (allInputsBool(node, 2)) {
    ERROR_ON(!hasUnityValue(alpha_param));
    return createLogical_or(graph, {node->input(0), node->input(1)});
  }

  // Ordinary addition
  torch::jit::Value *alpha_multiplicand = node->input(1);
  if (!hasUnityValue(alpha_param)) {
    auto alpha_node = createMul(graph, {alpha_param, alpha_multiplicand});
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
  auto x = node->input(0);
  auto min_node = node->input(1)->node();
  auto min = isNone(min_node) ? std::numeric_limits<float>::lowest()
                              : constantToFloat(min_node);

  auto max_node = node->input(2)->node();
  auto max = isNone(max_node) ? std::numeric_limits<float>::max()
                              : constantToFloat(max_node);
  return createClip(graph, {x}, max, min);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::add, addHandler);
  registerHandler(c10::aten::trunc, truncHandler);
  registerHandler(c10::aten::frac, fracHandler);
  registerHandler(c10::aten::floor_divide, floorDivideHandler);
  registerHandler(c10::aten::mul, mulHandler);
  registerHandler(c10::aten::true_divide, trueDivideHandler);
  registerHandler(c10::aten::clamp, clampHandler);
  registerHandler(c10::aten::clamp_, clampHandler);
}

} // namespace poptorch
