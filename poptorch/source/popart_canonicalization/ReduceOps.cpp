// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

#include "../PoptorchSymbols.hpp"

namespace poptorch {
namespace {

torch::jit::Node *reduceHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // Reductions have two overloads. The first is:
  // aten::mean(Tensor self, int[] dim, int keepdim, Tensor? out)) -> tensor

  // The second is:
  // aten::mean(Tensor self, int? dtype)) -> tensor

  torch::jit::Symbol kind = node->kind();
  torch::jit::Value *input = node->input(0);

  std::vector<std::int64_t> axes{};
  std::int64_t keepdim = 0;

  // Case 2.
  if (node->inputs().size() == 2) {
    torch::jit::Node *flatten = createFlatten(graph, {node->input(0)}, 0);
    input = flatten->output();
    axes = {1};
  } else {
    // Case 1.
    // Sometimes the dimensions are just one int.

    if (node->input(1)->node()->kind() == symbols::poptorch::tensor_constant) {
      axes.push_back(constantToLong(node->input(1)->node()));
    } else {
      axes = constantToLongVec(node->input(1)->node());
    }

    keepdim = constantToLong(node->input(2)->node());
  }

  // Output the correct reduction.
  if (kind == c10::aten::prod) {
    return createReduceprod(graph, {input}, axes, keepdim);
  }
  if (kind == c10::aten::mean) {
    return createReducemean(graph, {input}, axes, keepdim);
  }
  if (kind == c10::aten::sum) {
    return createReducesum(graph, {input}, axes, keepdim);
  }
  if (kind == c10::aten::logsumexp) {
    return createReducelogsumexp(graph, {input}, axes, keepdim);
  }
  ERROR("Popart Canonicalisation: UNREACHABLE reached in reductions.");
}

torch::jit::Node *argMinMaxHandler(torch::jit::Graph *graph,
                                   torch::jit::Node *node) {
  //  aten::argmin(Tensor in, int? dim, int keep_dims) -> Tensor
  //  aten::argmax(Tensor in, int? dim, int keep_dims) -> Tensor
  // dim (int) – the dimension to reduce. If None, the argmax
  //             of the flattened input is returned.

  torch::jit::Symbol kind = node->kind();
  torch::jit::Value *input = node->input(0);

  std::optional<std::int64_t> dim;
  if (node->input(1)->node()->kind() == symbols::poptorch::tensor_constant) {
    dim = constantToLong(node->input(1)->node());
  }

  std::int64_t keep_dim = constantToLong(node->input(2)->node());

  // If dim is not provided we will flatten input so just use 0 in that
  // case.
  std::int64_t dim_to_use = 1;

  // Check if dim is NONE.
  if (!dim) {
    torch::jit::Node *flatten = createFlatten(graph, {node->input(0)}, 0);
    input = flatten->output();
  } else {
    dim_to_use = *dim;
  }

  // Create the actual argmax/argmin.
  if (kind == c10::aten::argmax) {
    return createArgmax(graph, {input}, dim_to_use, keep_dim);
  }
  return createArgmin(graph, {input}, dim_to_use, keep_dim);
}
} // namespace

// clang-format off
static bool handlers = registerHandlers(
    c10::aten::argmax, argMinMaxHandler,
    c10::aten::argmin, argMinMaxHandler,
    c10::aten::prod, reduceHandler,
    c10::aten::mean, reduceHandler,
    c10::aten::sum, reduceHandler,
    c10::aten::logsumexp, reduceHandler);
// clang-format on

} // namespace poptorch
