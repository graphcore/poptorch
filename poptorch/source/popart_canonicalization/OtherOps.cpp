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

torch::jit::Node *meshgridHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  // aten::meshgrid(Tensor[] tensors) -> Tensor[]

  std::vector<torch::jit::Value *> tensors =
      handleTensorList(node->input(0)->node());

  std::vector<std::int64_t> expand_shape;
  expand_shape.reserve(tensors.size());
  for (torch::jit::Value *tensor : tensors) {
    // Each tensor is 1D so the shape is just the first dim
    expand_shape.push_back(shapeFromTensor(tensor)[0]);
  }

  std::vector<torch::jit::Value *> grids;
  for (std::size_t i = 0; i < tensors.size(); i++) {
    std::vector<std::int64_t> shape(tensors.size(), 1);
    shape[i] = -1;
    // Reshape 1D tensor to rank N, N = number of tensors, such that
    // all but the ith dimension is a singleton
    torch::jit::Node *reshaped = createReshape(graph, tensors[i], shape);
    // Expand over the dimensions of all other tensors
    torch::jit::Node *expanded =
        createExpand(graph, {reshaped->output(),
                             intVectorToIrConstant(graph, expand_shape)});
    grids.push_back(expanded->output());
  }

  return createAndInsertNode(graph, at::prim::ListConstruct, grids);
}
} // namespace

// clang-format off
static bool handlers = registerHandlers(
    c10::aten::einsum, einsumHandler,
    c10::aten::meshgrid, meshgridHandler);
// clang-format on

} // namespace poptorch
