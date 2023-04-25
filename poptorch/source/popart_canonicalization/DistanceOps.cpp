// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <ATen/ExpandUtils.h>

#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *pairwiseDistanceHandler(torch::jit::Graph *graph,
                                          torch::jit::Node *node) {
  // aten::pairwise_distance(Tensor x1, Tensor x2, float p, float eps,
  //                         bool keepdim)

  // Input 1
  auto *x1 = node->input(0);
  // Input 2
  auto *const x2 = node->input(1);
  // Norm degree
  auto *const p = node->input(2);
  // Small value to avoid division by zero
  auto *const eps = node->input(3);
  // Whether to keep vector dimension
  auto *const keepdim = node->input(4);
  auto input_shape = shapeFromTensor(x1);
  bool reshape_output = false;

  // No batch dim, append one to front
  // (D) -> (N, D), N = 1
  if (input_shape.size() == 1) {
    input_shape = {1, input_shape[0]};
    x1 = createUnsqueeze(graph, {x1}, {0})->output();
    reshape_output = true;
  }

  // x1 - x2
  auto *const x1_minus_x2 = createSub(graph, {x1, x2})->output();
  // x1 - x2 + eps
  auto *const x1_minus_x2_plus_eps =
      createAdd(graph, {x1_minus_x2, eps})->output();
  x1_minus_x2_plus_eps->setType(
      x1_minus_x2_plus_eps->type()->expect<c10::TensorType>()->withSizes(
          input_shape));

  // 1
  auto *const ones = wrapInConstant1D(graph, 1);
  // tensorNormHandler expects ListConstruct for dims
  torch::jit::Node *const ones_list =
      createAndInsertNode(graph, c10::prim::ListConstruct, {ones});

  std::vector<torch::jit::Value *> norm_inputs = {x1_minus_x2_plus_eps, p,
                                                  ones_list->output(), keepdim};
  // norm(x1 - x2 + eps, p, 1, keepdim)
  auto *out =
      createHandlerOperation(graph, getHandler(c10::aten::norm), norm_inputs);

  // If passed inputs of size (1, N), the output of norm will have shape
  // torch.Size([1]), but torch outputs torch.Size([]), so reshape
  if (reshape_output) {
    out = createReshape(graph, out->output(), shapeFromTensor(node->output(0)));
  }

  return out;
}

torch::jit::Node *cosineSimilarityHandler(torch::jit::Graph *graph,
                                          torch::jit::Node *node) {
  // aten::cosine_similarity(const Tensor& x1, const Tensor& x2, int64_t dim,
  //                         double eps)

  // inputs
  auto *const x1 = node->input(0);
  auto *const x2 = node->input(1);
  const auto dim = constantToLong(node->input(2)->node());
  auto *const eps = node->input(3);

  // dividend
  auto *const mul12 = createMul(graph, {x1, x2})->output();
  auto *const dot12 = createReducesum(graph, {mul12}, {dim}, 0)->output();

  // divisor
  auto *const mag1_sq = createReducesumsquare(graph, {x1}, {dim}, 0)->output();
  auto *const mag2_sq = createReducesumsquare(graph, {x2}, {dim}, 0)->output();
  auto *const mag12_sq = createMul(graph, {mag1_sq, mag2_sq})->output();
  auto *const mag12 = createSqrt(graph, {mag12_sq})->output();
  auto *const mag12_nonzero = createMax(graph, {mag12, eps})->output();

  return createDiv(graph, {dot12, mag12_nonzero});
}

torch::jit::Node *cdistHandler(torch::jit::Graph *graph,
                               torch::jit::Node *node) {
  // Input 1
  auto *const x1 = node->input(0);
  // Input 2
  auto *const x2 = node->input(1);
  // Norm degree
  auto *const p_degree = node->input(2);

  const std::vector<std::int64_t> x1_shape = shapeFromTensor(x1);
  const std::vector<std::int64_t> x2_shape = shapeFromTensor(x2);

  const auto ndim_x1 = x1_shape.size();
  const auto ndim_x2 = x2_shape.size();

  std::vector<std::int64_t> x1_shape_expanded;
  std::vector<std::int64_t> x2_shape_expanded;

  if (ndim_x1 > 0) {
    const auto m = x1_shape.at(ndim_x1 - 1);
    x1_shape_expanded.push_back(m);
  }
  if (ndim_x2 > 0) {
    const auto m = x2_shape.at(ndim_x2 - 1);
    x2_shape_expanded.push_back(m);
  }

  if (ndim_x1 > 1) {
    const auto p = x1_shape.at(ndim_x1 - 2);
    x1_shape_expanded.insert(x1_shape_expanded.begin(), {p, 1});
  }

  if (ndim_x2 > 1) {
    const auto r = x2_shape.at(ndim_x2 - 2);
    x2_shape_expanded.insert(x2_shape_expanded.begin(), {1, r});
  }

  std::vector<std::int64_t> b_x1;
  std::vector<std::int64_t> b_x2;

  if (ndim_x1 > 2) {
    b_x1 = {x1_shape.begin(), x1_shape.end() - 2};
  }

  if (ndim_x2 > 2) {
    b_x2 = {x2_shape.begin(), x2_shape.end() - 2};
  }

  if (b_x1 != b_x2) {
    const auto get_broadcasted_batch_shape =
        [](const std::vector<int64_t> &batch_shape,
           const std::vector<int64_t> &inferred_size) {
          if (batch_shape == inferred_size) {
            return batch_shape;
          }

          std::vector<std::int64_t> broadcasted_shape;

          const auto batch_shape_size = batch_shape.size();

          std::for_each(
              inferred_size.crbegin(), inferred_size.crend(),
              [cnt = 0u, batch_shape_size, &broadcasted_shape,
               &batch_shape](const auto &inferred_value) mutable {
                if (cnt >= batch_shape_size) {
                  broadcasted_shape.insert(broadcasted_shape.begin(), 1);
                  return;
                }

                const auto batch_shape_value =
                    batch_shape.at(batch_shape_size - cnt - 1);

                if (inferred_value != batch_shape_value &&
                    batch_shape_value != 1) {
                  broadcasted_shape.insert(broadcasted_shape.begin(), 1);
                } else {
                  broadcasted_shape.insert(broadcasted_shape.begin(),
                                           batch_shape_value);
                  ++cnt;
                }
              });

          return broadcasted_shape;
        };

    const std::vector<int64_t> expand_batch_portion =
        at::infer_size(b_x1, b_x2);

    b_x1 = get_broadcasted_batch_shape(b_x1, expand_batch_portion);
    b_x2 = get_broadcasted_batch_shape(b_x2, expand_batch_portion);
  }

  x1_shape_expanded.insert(x1_shape_expanded.begin(), b_x1.cbegin(),
                           b_x1.cend());
  x2_shape_expanded.insert(x2_shape_expanded.begin(), b_x2.cbegin(),
                           b_x2.cend());

  auto *x1_expanded = createReshape(graph, x1, x1_shape_expanded)->output();
  auto *x2_expanded = createReshape(graph, x2, x2_shape_expanded)->output();

  auto *x1_minus_x2 = createSub(graph, {x1_expanded, x2_expanded})->output();
  auto *dims = createAndInsertNode(graph, c10::prim::ListConstruct,
                                   {wrapInConstant1D(graph, -1)})
                   ->output();
  auto *keepdim = createConstantLong(graph, {0}, {1})->output();

  return createHandlerOperation(graph, getHandler(c10::aten::norm),
                                {x1_minus_x2, p_degree, dims, keepdim});
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::pairwise_distance, pairwiseDistanceHandler);
  registerHandler(c10::aten::cosine_similarity, cosineSimilarityHandler);
  registerHandler(c10::aten::cdist, cdistHandler);
  registerHandler(c10::aten::_cdist_forward, cdistHandler);
}
} // namespace poptorch
