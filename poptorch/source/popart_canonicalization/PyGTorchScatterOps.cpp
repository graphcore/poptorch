// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "../PoptorchSymbols.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {
namespace {
std::int32_t getReductionMethod(torch::jit::Node *node) {
  auto kind = node->kind();
  if (kind == torch_scatter::scatter_max) {
    return 1;
  }
  if (kind == torch_scatter::scatter_min) {
    return 2;
  }
  ERROR("Unsupported reduction for node: " << nodeToString(node));
}

torch::jit::Node *scatterMaxMinHandler(torch::jit::Graph *graph,
                                       torch::jit::Node *node) {
  // Signatures for scatter_max and scatter_min:
  // (Tensor src, Tensor index, int dim, Tensor? out, int? dim_size)
  auto *src = node->input(0);
  auto *index = node->input(1);
  auto t = src->type()->expect<c10::TensorType>();
  auto axis = handleDimensionParam(node->input(2), t);

  auto *opt_out = node->input(3);
  ERROR_ON_MSG(!isNone(opt_out),
               "Providing the optional output is not currently supported.");

  // Both scatter_max and scatter_min return two outputs but we only support the
  // first output so we delete the second one to match the popart output
  node->eraseOutput(1);

  auto shape = shapeFromTensor(node->output());
  auto axissize = shape.at(axis);

  auto *opt_axissize = node->input(4);
  if (!isNone(opt_axissize)) {
    axissize = constantToInt(opt_axissize->node());
  }

  return createScatterreduce(graph, {src, index}, axissize, axis,
                             getReductionMethod(node));
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(torch_scatter::scatter_max, scatterMaxMinHandler);
  registerHandler(torch_scatter::scatter_min, scatterMaxMinHandler);
}

} // namespace poptorch
