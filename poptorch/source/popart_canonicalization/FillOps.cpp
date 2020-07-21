
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *maskedFillHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // Derived from documentation
  // aten::masked_fill(Tensor self, Tensor mask, Tensor other) -> Tensor

  // Apply by performing the following operation
  // inverseMask = -(mask - 1)
  // self * inverseMask + mask * other

  // Cast the mask to int32.
  torch::jit::Node *mask = createCast(graph, node->input(1), c10::kInt);

  // Create an inverse mask via -(mask - 1)
  torch::jit::Node *negative_one = createConstantInt(graph, {-1}, {1});

  torch::jit::Node *inverse_mask =
      createAdd(graph, {mask->output(), negative_one->output()});

  inverse_mask = createNeg(graph, {inverse_mask->output()});

  // Prepare input and update
  mask = createCast(graph, node->input(1), c10::kFloat);

  float other_as_const = *handleConstant<float>(node->input(2)->node());
  torch::jit::Node *other = createConstantFloat(graph, {other_as_const}, {1});

  torch::jit::Node *update =
      createMul(graph, {mask->output(), other->output()});

  // Create holes in the original so we can add into it.
  inverse_mask = createCast(graph, inverse_mask->output(), c10::kFloat);

  torch::jit::Node *self =
      createMul(graph, {node->input(0), inverse_mask->output()});

  return createAdd(graph, {self->output(), update->output()});
}
} // namespace

// clang-format off
static bool handlers = registerHandlers(
    c10::aten::masked_fill, maskedFillHandler);
// clang-format on

} // namespace poptorch
