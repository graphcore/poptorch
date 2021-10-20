// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "PopartCanonicalizationUtils.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *dropoutHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  auto *x = node->input(0);
  auto p = constantToFloat(node->input(1)->node());
  auto train = constantToBool(node->input(2)->node());

  if (!train) {
    return createIdentity(graph, {x});
  }

  return createDropout(graph, {x}, 1, p);
}

torch::jit::Node *featureDropoutHandler(torch::jit::Graph *graph,
                                        torch::jit::Node *node) {
  torch::jit::Value *input = node->input(0);
  float ratio = constantToFloat(node->input(1)->node());
  bool train = constantToBool(node->input(2)->node());

  if (!train) {
    return createIdentity(graph, {input});
  }

  // Input tensor is required to be more than 2-d since feature dropout assumes
  // that the input represents a 2-d map of features: N x C x (feature shape)
  std::vector<int64_t> drop_shape = shapeFromTensor(input);
  ERROR_ON_MSG(drop_shape.size() < 2,
               "Feature dropout requires at least 2 dimensions in the input");

  // The dropout mask shape will be N x C with as many trailing singleton
  // dimensions as needed to meet the broadcast requirement
  std::fill(drop_shape.begin() + 2, drop_shape.end(), 1);

  return createShapeddropout(graph, {input}, drop_shape, ratio);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::feature_dropout, featureDropoutHandler);
  registerHandler(c10::aten::dropout, dropoutHandler);
}

} // namespace poptorch
