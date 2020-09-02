// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <limits>

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

torch::jit::Node *tensorNormHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // aten::norm(Tensor in, int p) -> Tensor
  // aten::norm(Tensor in, float p) -> Tensor
  // aten::norm(Tensor in, int p, int[] dim, int keepdim) -> Tensor
  // aten::norm(Tensor in, float p, int[] dim, int keepdim) -> Tensor
  torch::jit::Value *input = node->input(0);
  torch::jit::Value *p_val = node->input(1);

  std::vector<std::int64_t> axes{};
  std::int64_t keepdim = 0;

  if (node->inputs().size() == 2) {
    torch::jit::Node *flatten = createFlattenTypedOutput(graph, {input}, 0);
    input = flatten->output();
    axes = {1};
  } else {
    axes = constantToLongVec(node->input(2)->node());
    keepdim = constantToLong(node->input(3)->node());
  }

  constexpr float pos_inf = std::numeric_limits<float>::infinity();
  constexpr float neg_inf = -std::numeric_limits<float>::infinity();
  float p = constantToFloat(node->input(1)->node());

  if (p == 1.0) {
    return createReducel1(graph, {input}, axes, keepdim);
  }
  if (p == 2.0) {
    return createReducel2(graph, {input}, axes, keepdim);
  }
  if (p == pos_inf || p == neg_inf) {
    // max/min(abs(x))
    torch::jit::Node *abs = createAbs(graph, {input});
    input = abs->output();

    if (p == pos_inf) {
      return createReducemax(graph, {input}, axes, keepdim);
    }
    return createReducemin(graph, {input}, axes, keepdim);
  }

  // sum(abs(x)**p)**(1./p)
  torch::jit::Node *abs = createUnarySameTypedOutput(createAbs, graph, {input});

  torch::jit::Node *pow = createPow(graph, {abs->output(), p_val});
  torch::jit::Node *sum = createWithSameTypedOutput(
      createReducesum, graph, {pow->output()}, axes, keepdim);

  at::ScalarType p_type = getNodeScalarType(p_val);

  if (p_type == c10::ScalarType::Int || p_type == c10::ScalarType::Long) {
    // Cast int to float before reciprocal
    torch::jit::Node *to_float =
        createCastTypedOutput(graph, p_val, c10::kFloat);
    p_val = to_float->output();
  }

  torch::jit::Node *one_over_p =
      createUnarySameTypedOutput(createReciprocal, graph, {p_val});
  return createPow(graph, {sum->output(), one_over_p->output()});
}

} // namespace

// clang-format off
static bool handlers = registerHandlers(
    c10::aten::argmax, argMinMaxHandler,
    c10::aten::argmin, argMinMaxHandler,
    c10::aten::prod, reduceHandler,
    c10::aten::mean, reduceHandler,
    c10::aten::sum, reduceHandler,
    c10::aten::logsumexp, reduceHandler,
    c10::aten::norm, tensorNormHandler);
// clang-format on

} // namespace poptorch
