// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include <poptorch/OpBuilder.hpp>
#include <poptorch_logging/Error.hpp>

namespace poptorch {
namespace {
torch::jit::Node *expandHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::expand(Tensor self, int[] size)  -> Tensor
  torch::jit::Node *new_node;

  // Extract the type from the pytorch IR.
  c10::TensorTypePtr self_tensor =
      node->inputs()[0]->type()->expect<c10::TensorType>();
  c10::VaryingShape self_dims = self_tensor->sizes();

  // Old shape
  std::vector<std::int64_t> old_shape = shapeFromTensor(node->input(0));

  // Count the elems in the old shape.
  std::int64_t old_elem_count = std::accumulate(
      old_shape.begin(), old_shape.end(), 1, std::multiplies<std::int64_t>());

  // Get the target size for the expand.
  std::vector<std::int64_t> new_shape =
      handleList<int64_t>(node->input(1)->node());

  // Count the number of elements in the target shape.
  std::int64_t new_elem_count = std::accumulate(
      new_shape.begin(), new_shape.end(), 1, std::multiplies<std::int64_t>());

  // Elements don't change so just a reshape.
  if (new_elem_count == old_elem_count) {
    new_node = createReshape(graph, node->input(0), new_shape);
  } else {
    // Otherwise we are expanding the original tensor.
    new_node = createConstantInt(graph, new_shape,
                                 {static_cast<int64_t>(new_shape.size())});
    new_node = createCast(graph, new_node->output(), c10::kLong);
    new_node = createExpand(graph, {node->input(0), new_node->output()});
  }
  return new_node;
}

torch::jit::Node *reshapeHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  // aten::view(Tensor self, int[] size) -> Tensor
  // aten::unsqueeze(Tensor self, int dim) -> Tensor

  std::vector<std::int64_t> new_shape = shapeFromTensor(node->output());

  // Reshape the tensor into that shape.
  return createReshape(graph, node->inputs()[0], new_shape);
}

torch::jit::Node *expandAsHandler(torch::jit::Graph *graph,
                                  torch::jit::Node *node) {
  // aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor
  // aten::expand_as(Tensor self, Tensor other) -> Tensor
  torch::jit::Node *new_node;

  // Extract the type from the pytorch IR.
  c10::TensorTypePtr self_tensor =
      node->input(0)->type()->expect<c10::TensorType>();
  c10::VaryingShape self_dims = self_tensor->sizes();

  std::int64_t old_elem_count = 0;
  for (auto optional_int : *self_dims.sizes()) {
    old_elem_count += *optional_int;
  }

  // Extract the type from the pytorch IR.
  c10::TensorTypePtr as_tensor =
      node->input(1)->type()->expect<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  // Convert that IR type into a C++ vector of ints.
  std::vector<std::int64_t> new_shape;
  std::int64_t new_elem_count = 0;

  for (auto optional_int : *dims.sizes()) {
    new_shape.push_back(*optional_int);
    new_elem_count += *optional_int;
  }

  // Elements don't change so just a reshape.
  if (new_elem_count == old_elem_count) {
    new_node = createReshape(graph, node->input(0), new_shape);
  } else {
    new_node = createConstantInt(graph, new_shape,
                                 {static_cast<int64_t>(new_shape.size())});

    new_node = createCast(graph, new_node->output(), c10::kLong);

    new_node = createExpand(graph, {node->input(0), new_node->output()});
  }
  return new_node;
}
} // namespace

static bool handlers = registerHandlers(
    c10::aten::expand, expandHandler, c10::aten::expand_as, expandAsHandler,
    c10::aten::view, reshapeHandler, c10::aten::unsqueeze, reshapeHandler,
    c10::aten::flatten, reshapeHandler, c10::aten::reshape, reshapeHandler,
    c10::aten::squeeze, reshapeHandler);

} // namespace poptorch
