// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include <poptorch/OpBuilder.hpp>
#include <poptorch_logging/Error.hpp>
#include <poptorch_logging/Logging.hpp>

namespace poptorch {
namespace {
torch::jit::Node *softmaxHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  // "aten::softmax(Tensor self, int dim, int? dtype) -> Tensor"

  std::int64_t dim = *handleConstant<std::int64_t>(node->input(1)->node());

  c10::TensorTypePtr as_tensor =
      node->input(0)->type()->cast<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  if (dim < 0) {
    dim = *dims.size() + dim;
  }

  return createSoftmax(graph, {node->input(0)}, dim);
}

torch::jit::Node *logSoftmaxHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // "aten::log_softmax(Tensor self, int dim, int? dtype) -> Tensor"

  std::int64_t dim = *handleConstant<std::int64_t>(node->input(1)->node());

  c10::TensorTypePtr as_tensor =
      node->input(0)->type()->cast<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  if (dim < 0) {
    dim = *dims.size() + dim;
  }

  torch::jit::Node *softmax = createSoftmax(graph, {node->input(0)}, dim);

  return createLog(graph, {softmax->output()});
}
} // namespace

static bool handlers =
    registerHandlers(c10::aten::softmax, softmaxHandler, c10::aten::log_softmax,
                     logSoftmaxHandler);

} // namespace poptorch
