// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *pairwiseDistanceHandler(torch::jit::Graph *graph,
                                          torch::jit::Node *node) {
  // aten::pairwise_distance(Tensor x1, Tensor x2, float p, float eps,
  //                         bool keepdim)

  // Input 1
  torch::jit::Value *x1 = node->input(0);
  // Input 2
  torch::jit::Value *x2 = node->input(1);
  // Norm degree
  torch::jit::Value *p = node->input(2);
  // Small value to avoid division by zero
  torch::jit::Value *eps = node->input(3);
  // Whether to keep vector dimension
  torch::jit::Value *keepdim = node->input(4);

  // tensorNormHandler
  auto norm_handler = getHandler(c10::aten::norm);

  // x1 - x2
  torch::jit::Node *x1_minus_x2 = createSub(graph, {x1, x2});
  // x1 - x2 + eps
  torch::jit::Node *x1_minus_x2_plus_eps =
      createAdd(graph, {x1_minus_x2->output(), eps});
  // 1
  torch::jit::Node *ones = createConstantInt(graph, {1}, {});
  // tensorNormHandler expects ListConstruct for dims
  torch::jit::Node *ones_list =
      createAndInsertNode(graph, c10::prim::ListConstruct, {ones->output()});

  std::vector<torch::jit::Value *> norm_inputs = {
      x1_minus_x2_plus_eps->output(), p, ones_list->output(), keepdim};
  // norm(x1 - x2 + eps, p, 1, keepdim)
  return createHandlerOperation(graph, norm_handler, norm_inputs);
}
} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::pairwise_distance, pairwiseDistanceHandler);
}

} // namespace poptorch
