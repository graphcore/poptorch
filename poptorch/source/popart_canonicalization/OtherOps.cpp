// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "EinsumOp.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {
torch::jit::Node *einsumHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::einsum(string equation, Tensor[] tensors) -> Tensor

  // Einstein summation convention equation
  std::string eq = constantToString(node->input(0)->node());
  // List of inputs to perform the operation on
  std::vector<torch::jit::Value *> tensors =
      handleTensorList(node->input(1)->node());

  EinsumOp einsum(eq, tensors);
  return einsum.create(graph);
}
} // namespace

// clang-format off
static bool handlers = registerHandlers(
    c10::aten::einsum, einsumHandler);
// clang-format on

} // namespace poptorch
