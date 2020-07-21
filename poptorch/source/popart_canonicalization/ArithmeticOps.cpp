// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *rsqrtHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // rsqrt =  1 / sqrt(x)
  torch::jit::Node *sqrt = createSqrt(graph, {node->input()});

  return createReciprocal(graph, {sqrt->output()});
}

torch::jit::Node *expm1Handler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // expm1 = exp(x) - 1

  // exp(x)
  torch::jit::Node *exp = createExp(graph, {node->input()});

  // Add the one constant
  torch::jit::Node *one = createConstantFloat(graph, {1.0}, {});

  return createSub(graph, {exp->output(), one->output()});
}

torch::jit::Node *truncHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // Drop the exponent by casting to int and back.
  torch::jit::Node *to_int = createCast(graph, node->input(), c10::kInt);

  return createCast(graph, to_int->output(), c10::kFloat);
}

torch::jit::Node *fracHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // Frac(x) = x - trunc(x)

  // Drop the exponent by casting to int and back.
  torch::jit::Node *to_int = createCast(graph, node->input(), c10::kInt);

  torch::jit::Node *trunc = createCast(graph, to_int->output(), c10::kFloat);

  return createSub(graph, {node->input(), trunc->output()});
}

torch::jit::Node *roundHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // round(x) = trunc(x + sign(x)*0.5)

  // Add 0.5 as constant.
  torch::jit::Node *zero_point_five = createConstantFloat(graph, {0.5}, {});

  torch::jit::Node *sign = createSign(graph, {node->input()});

  torch::jit::Node *broadcast_by_sign =
      createMul(graph, {sign->output(), zero_point_five->output()});

  torch::jit::Node *addition =
      createAdd(graph, {node->input(), broadcast_by_sign->output()});

  // Drop the exponent by casting to int and back.
  torch::jit::Node *to_int = createCast(graph, addition->output(), c10::kInt);

  return createCast(graph, to_int->output(), c10::kFloat);
}

torch::jit::Node *floorDivideHandler(torch::jit::Graph *graph,
                                     torch::jit::Node *node) {
  // aten::floor_divide(Tensor x, Tensor y) -> Tensor
  // floor_divide(x, y) = floor(x)/floor(y)

  torch::jit::Node *x = createFloor(graph, {node->inputs()[0]});
  torch::jit::Node *y = createFloor(graph, {node->inputs()[1]});

  return createDiv(graph, {x->output(), y->output()});
}

torch::jit::Node *trueDivideHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // aten::true_divide(Tensor x, Tensor y) -> Tensor
  // true_divide(x, y) = (float)x / (float)y

  torch::jit::Node *x = createCast(graph, node->inputs()[0], c10::kFloat);

  torch::jit::Node *y = createCast(graph, node->inputs()[1], c10::kFloat);

  return createDiv(graph, {x->output(), y->output()});
}

torch::jit::Node *rsubHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // Tensor aten::rsub(const Tensor& self, const Tensor& other, Scalar alpha)
  // We are ignoring alpha here.

  torch::jit::Value *other = node->input(1);

  std::optional<float> as_scalar = handleConstant<float>(other->node());

  // This operation can also take a scalar for other. If it is that overload
  // then we have to add it as a popart scalar and work with that instead.
  if (as_scalar) {
    torch::jit::Node *as_constant =
        createConstantFloat(graph, {*as_scalar}, {1});

    other->replaceAllUsesWith(as_constant->output());

    // Mark it for deletion.
    markNodeForDeletion(other->node());

    // Use the popart constant instead.
    other = as_constant->output();
  }

  return createSub(graph, {other, node->input(0)});
}
} // namespace

// clang-format off
static bool handlers = registerHandlers(
    c10::aten::rsqrt, rsqrtHandler,
    c10::aten::rsub, rsubHandler,
    c10::aten::expm1, expm1Handler,
    c10::aten::trunc, truncHandler,
    c10::aten::frac, fracHandler,
    c10::aten::round, roundHandler,
    c10::aten::floor_divide, floorDivideHandler,
    c10::aten::true_divide, trueDivideHandler);
// clang-format on

} // namespace poptorch
