// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {
torch::jit::Node *sizeHandler(torch::jit::Graph *graph,
                              torch::jit::Node *node) {
  //  aten::size(Tensor input, int dim) -> int
  std::vector<std::int64_t> shape = shapeFromTensor(node->input(0));
  std::int64_t dim = *handleConstant<std::int64_t>(node->input(1)->node());
  return createConstantInt(graph, {shape[dim]}, {1});
}

torch::jit::Node *numToTensorHandler(torch::jit::Graph *graph,
                                     torch::jit::Node *node) {
  return createIRConstant(graph, node->input(0));
}
} // namespace

static bool handlers = registerHandlers(
    c10::aten::size, sizeHandler, c10::prim::NumToTensor, numToTensorHandler);

} // namespace poptorch
