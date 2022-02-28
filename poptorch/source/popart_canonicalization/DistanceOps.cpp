// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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
  auto *x2 = node->input(1);
  // Norm degree
  auto *p = node->input(2);
  // Small value to avoid division by zero
  auto *eps = node->input(3);
  // Whether to keep vector dimension
  auto *keepdim = node->input(4);
  auto input_shape = shapeFromTensor(x1);
  bool reshape_output = false;

  // No batch dim, append one to front
  // (D) -> (N, D), N = 1
  if (input_shape.size() == 1) {
    input_shape = {1, input_shape[0]};
    x1 = createUnsqueeze(graph, {x1}, {0})->output();
    reshape_output = true;
  }

  // tensorNormHandler
  auto norm_handler = getHandler(c10::aten::norm);

  // x1 - x2
  auto *x1_minus_x2 = createSub(graph, {x1, x2})->output();
  // x1 - x2 + eps
  auto *x1_minus_x2_plus_eps = createAdd(graph, {x1_minus_x2, eps})->output();
  x1_minus_x2_plus_eps->setType(
      x1_minus_x2_plus_eps->type()->expect<c10::TensorType>()->withSizes(
          input_shape));

  // 1
  auto *ones = wrapInConstant1D(graph, 1);
  // tensorNormHandler expects ListConstruct for dims
  torch::jit::Node *ones_list =
      createAndInsertNode(graph, c10::prim::ListConstruct, {ones});

  std::vector<torch::jit::Value *> norm_inputs = {x1_minus_x2_plus_eps, p,
                                                  ones_list->output(), keepdim};
  // norm(x1 - x2 + eps, p, 1, keepdim)
  auto *out = createHandlerOperation(graph, norm_handler, norm_inputs);

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
  auto *x1 = node->input(0);
  auto *x2 = node->input(1);
  auto dim = constantToLong(node->input(2)->node());
  auto *eps = node->input(3);

  // dividend
  auto *mul12 = createMul(graph, {x1, x2})->output();
  auto *dot12 = createReducesum(graph, {mul12}, {dim}, 0)->output();

  // divisor
  auto *mag1_sq = createReducesumsquare(graph, {x1}, {dim}, 0)->output();
  auto *mag2_sq = createReducesumsquare(graph, {x2}, {dim}, 0)->output();
  auto *mag12_sq = createMul(graph, {mag1_sq, mag2_sq})->output();
  auto *mag12 = createSqrt(graph, {mag12_sq})->output();
  auto *mag12_nonzero = createMax(graph, {mag12, eps})->output();

  return createDiv(graph, {dot12, mag12_nonzero});
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::pairwise_distance, pairwiseDistanceHandler);
  registerHandler(c10::aten::cosine_similarity, cosineSimilarityHandler);
}

} // namespace poptorch
