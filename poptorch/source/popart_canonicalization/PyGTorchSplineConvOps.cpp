// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {

torch::jit::Node *torchSplineBasisHandler(torch::jit::Graph *graph,
                                          torch::jit::Node *node) {
  // Signatures for spline_basis
  // (Tensor pseudo, Tensor kernelSize, Tensor isOpenSpline, int degree)

  const std::vector<torch::jit::Value *> args{node->input(0), node->input(1),
                                              node->input(2)};
  const std::int32_t degree = constantToInt(node->input(3)->node());

  auto *result = createSplinebasis(graph, args, degree);

  return result;
}

torch::jit::Node *torchSplineWeightingHandler(torch::jit::Graph *graph,
                                              torch::jit::Node *node) {
  // Signatures for spline_weighting
  // (Tensor input, Tensor weight, Tensor basis, Tensor weightIndex)

  const std::vector<torch::jit::Value *> args{node->input(0), node->input(1),
                                              node->input(2), node->input(3)};

  auto *result = createSplineweighting(graph, args);

  return result;
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(torch_spline_conv::spline_basis, torchSplineBasisHandler);
  registerHandler(torch_spline_conv::spline_weighting,
                  torchSplineWeightingHandler);
}

} // namespace poptorch
