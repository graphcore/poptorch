// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {
template <typename SoftmaxFunc>
torch::jit::Node *handleSoftmaxOp(torch::jit::Graph *graph,
                                  torch::jit::Node *node,
                                  SoftmaxFunc &&softmax_fn) {
  std::vector<int64_t> input_shape = shapeFromTensor(node->input(0));
  int64_t rank = static_cast<int64_t>(input_shape.size());
  std::int64_t dim = constantToLong(node->input(1)->node());

  if (dim < 0) {
    dim = rank + dim;
  }

  if (rank < 2 || dim == rank - 1) {
    return softmax_fn(graph, {node->input(0)}, dim);
  }

  // ONNX (log)softmax up to version 13 specifies that the input is
  // coerced to 2D where the axis attribute demarcates the flattening dim.
  // To workaround this we:
  //
  // 1. permute the dim arg to the final dimension
  // 2. evaluate (log)softmax using last dim as the axis
  // 3. permute result back to the original dimension order.
  //
  // Opset 13 brings the ONNX spec in line with the interpretation of the dim
  // argument as implemented by torch so this may need updating when popart
  // adds support for opset 13.
  std::vector<std::int64_t> perm(rank);
  std::iota(perm.begin(), perm.end(), 0);
  std::swap(perm[dim], perm.back());
  torch::jit::Node *transpose = createTranspose(graph, {node->input(0)}, perm);
  torch::jit::Node *sm = softmax_fn(graph, {transpose->output()}, rank - 1);
  return createTranspose(graph, {sm->output()}, perm);
}

torch::jit::Node *softmaxHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  // "aten::softmax(Tensor self, int dim, int? dtype) -> Tensor"
  return handleSoftmaxOp(graph, node, createSoftmax);
}

torch::jit::Node *logSoftmaxHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // "aten::log_softmax(Tensor self, int dim, int? dtype) -> Tensor"
  return handleSoftmaxOp(graph, node, createLogsoftmax);
}
} // namespace

// clang-format off
static bool handlers =
    registerHandlers(
        c10::aten::softmax, softmaxHandler,
        c10::aten::log_softmax, logSoftmaxHandler);
// clang-format on

} // namespace poptorch
