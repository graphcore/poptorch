// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <torch/csrc/jit/ir/ir.h>

#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"

namespace poptorch {
namespace {

torch::jit::Node *sliceHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor
  // // NOLINT

  std::int64_t dim = constantToLong(node->input(1)->node());

  std::int64_t start = constantToLong(node->input(2)->node());

  std::int64_t end = constantToLong(node->input(3)->node());

  c10::TensorTypePtr as_tensor =
      node->input(0)->type()->cast<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  // Based on aten/src/ATen/native/TensorShape.cpp slice()
  if (start < 0) {
    start += *dims[dim];
  }
  if (end < 0) {
    end += *dims[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= *dims[dim]) {
    start = *dims[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= *dims[dim]) {
    end = *dims[dim];
  }

  return createSlice(graph, {node->input(0), wrapInConstant1D(graph, start),
                             wrapInConstant1D(graph, end),
                             wrapInConstant1D(graph, dim)});
}

} // namespace

// clang-format off
static bool handlers = registerHandlers(
    c10::aten::slice,  sliceHandler);
// clang-format on

} // namespace poptorch
