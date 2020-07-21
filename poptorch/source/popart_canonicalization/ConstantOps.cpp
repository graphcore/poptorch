
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *onesHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  // aten::ones(int[] size, *, int? dtype, int? layout, Device? device, bool?
  // pin_memory) -> Tensor
  c10::TensorTypePtr as_tensor =
      node->outputs()[0]->type()->expect<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();
  std::vector<std::int64_t> operation_shape;

  for (auto optional_int : *dims.sizes()) {
    operation_shape.push_back(*optional_int);
  }

  switch (*as_tensor->scalarType()) {
  case c10::ScalarType::Int:
  case c10::ScalarType::Long: {
    return createConstantInt(graph, {1}, operation_shape);
    break;
  }
  case c10::ScalarType::Float: {
    return createConstantFloat(graph, {1.0}, operation_shape);
    break;
  }
  default:
    ERROR("aten::ones of type " << c10::toString(*as_tensor->scalarType())
                                << " not supported");
  }
}

torch::jit::Node *zerosHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // aten::zeros(int[] size, *, int? dtype, int? layout, Device? device, bool?
  // pin_memory) -> Tensor
  c10::TensorTypePtr as_tensor =
      node->outputs()[0]->type()->expect<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();
  std::vector<std::int64_t> operation_shape;

  for (auto optional_int : *dims.sizes()) {
    operation_shape.push_back(*optional_int);
  }

  switch (*as_tensor->scalarType()) {
  case c10::ScalarType::Int:
  case c10::ScalarType::Long: {
    return createConstantInt(graph, {0}, operation_shape);
  }
  case c10::ScalarType::Float: {
    return createConstantFloat(graph, {0.0}, operation_shape);
    break;
  }
  default:
    ERROR("aten::zeros of type " << c10::toString(*as_tensor->scalarType())
                                 << " not supported");
  }
}

torch::jit::Node *arangeHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
  // aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype,
  // Layout, Device, bool pin_memory)

  ERROR_ON_MSG(node->inputs().size() != 5, "Unsupported arrange op");

  std::vector<std::int64_t> vals;
  std::size_t end = *handleConstant<std::int64_t>(node->input(0)->node());
  for (std::size_t start = 0; start < end; ++start) {
    vals.push_back(start);
  }

  return createConstantInt(graph, vals,
                           {static_cast<std::int64_t>(vals.size())});
}

} // namespace

// clang-format off
static bool handlers = registerHandlers(
    c10::aten::arange, arangeHandler,
    c10::aten::ones, onesHandler,
    c10::aten::zeros, zerosHandler);
// clang-format on

} // namespace poptorch
