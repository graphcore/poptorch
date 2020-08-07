// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "../PoptorchSymbols.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {

torch::jit::Node *normalHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::normal(float mean, float std, int[] size, Generator?, int? dtype,
  // int? layout, Device? device, bool? pin_memory) -> Tensor
  std::vector<int64_t> shape = shapeFromTensor(node->output());
  std::optional<float> mean = handleConstant<float>(node->input(0)->node());
  std::optional<float> scale = handleConstant<float>(node->input(1)->node());
  ERROR_ON_MSG(!mean || !scale, "Invalid input arguments. Expected scalar "
                                "float values for both mean and std.");
  return createRandomNormal(graph, shape, *mean, *scale);
}

} // namespace

static bool handler = registerHandlers(c10::aten::normal, normalHandler);

} // namespace poptorch
