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

  ERROR_ON_MSG(node->input(0)->node()->kind() !=
                   symbols::poptorch::tensor_constant,
               "random normal is only supported with a scalar mean");

  ERROR_ON_MSG(
      node->input(1)->node()->kind() != symbols::poptorch::tensor_constant,
      "random normal is only supported with a scalar standard deviation");

  std::vector<int64_t> shape = shapeFromTensor(node->output());
  float mean = constantToFloat(node->input(0)->node());
  float scale = constantToFloat(node->input(1)->node());

  return createRandomNormal(graph, shape, mean, scale);
}

} // namespace

static bool handler = registerHandlers(c10::aten::normal, normalHandler);

} // namespace poptorch
