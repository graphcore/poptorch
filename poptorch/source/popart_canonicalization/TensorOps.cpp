// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {
namespace {
torch::jit::Node *sizeHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  //  aten::size(Tensor input, int dim) -> int
  std::vector<std::int64_t> shape = shapeFromTensor(node->input(0));
  std::int64_t dim = constantToLong(node->input(1)->node());
  return createConstantInt(graph, {shape[dim]}, {1});
}

torch::jit::Node *numToTensorHandler(torch::jit::Graph *graph,
                                     torch::jit::Node *node) {
  // Should be a tensor already
  ERROR_ON(node->input(0)->node()->kind() !=
           symbols::poptorch::tensor_constant);
  UNUSED(graph);
  node->output()->replaceAllUsesWith(node->input(0));
  markNodeForDeletion(node);
  return nullptr;
}

// Input tensor of shape [M, N, ...] is repeated in [R1, R2, ...]
// dimensions by:
//   1) transforming to [1, M, 1, N, ...]
//   2) expanding to [R1, M, R2, N, ...]
//   3) reshaping to [R1*M, R2*N, ...]
torch::jit::Node *repeatHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  std::vector<std::int64_t> old_shape = shapeFromTensor(node->input(0));
  std::vector<std::int64_t> new_shape = shapeFromTensor(node->output());
  std::vector<std::int64_t> dim_repeats =
      constantToLongVec(node->input(1)->node());
  std::vector<std::int64_t> dim_expands;
  std::vector<std::int64_t> transform_shape;

  // If repeat dimensions exceed shape dimensions, pad the front of the
  // original shape with singleton dimensions so that it can
  // be expanded
  size_t padding = dim_repeats.size() > old_shape.size()
                       ? dim_repeats.size() - old_shape.size()
                       : 0;

  torch::jit::Node *new_node = node->input(0)->node();

  for (std::size_t i = 0; i < dim_repeats.size(); i++) {
    dim_expands.push_back(dim_repeats[i]);

    std::int64_t padded_dim = i < padding ? 1 : old_shape[i - padding];
    if (padded_dim > 1 && dim_repeats[i] > 1) {
      transform_shape.push_back(1);
      dim_expands.push_back(padded_dim);
    }
    transform_shape.push_back(padded_dim);
  }

  new_node = createReshape(graph, new_node->output(), transform_shape);
  new_node = createExpand(
      graph, {new_node->output(), intVectorToIrConstant(graph, dim_expands)});

  return createReshape(graph, new_node->output(), new_shape);
}

torch::jit::Node *copy_Handler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // aten::copy_(Tensor self, Tensor src, bool non_blocking) -> Tensor
  at::ScalarType dest_type = getNodeScalarType(node->input(0));

  return createCast(graph, node->input(1), dest_type);
}
} // namespace

static bool handlers = registerHandlers(
    c10::aten::size, sizeHandler,
    c10::prim::NumToTensor, numToTensorHandler,
    c10::aten::repeat, repeatHandler,
    c10::aten::copy_, copy_Handler);
} // namespace poptorch
