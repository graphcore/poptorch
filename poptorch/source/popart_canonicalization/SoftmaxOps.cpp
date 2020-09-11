// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {
torch::jit::Node *softmaxHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  // "aten::softmax(Tensor self, int dim, int? dtype) -> Tensor"

  std::int64_t dim = constantToLong(node->input(1)->node());

  c10::TensorTypePtr as_tensor =
      node->input(0)->type()->cast<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  int64_t dimensions = *dims.size();

  if (dim < 0) {
    dim = dimensions + dim;
  }

  if (dimensions >= 2 && dim != dimensions - 1) {
    // Handle an input of at least 2D.
    // Transpose the "dim" of the input to the last dimension,
    // and revert to the original shape after done with softmax.
    // The calling sequence is: transpose, softmax, and transpose.
    //
    // For example, for an 4D input tensor, the IRs are like below:
    // graph(%input : Float(11, 22, 33, 44)):
    // %8 : Tensor = popart::transpose[perm=[0, 2, 3, 1]](%input)
    // %9 : Tensor = popart::softmax[axis=3](%8)
    // %10 : Tensor = popart::transpose[perm=[0, 3, 1, 2]](%9)

    std::vector<std::int64_t> before_transpose;
    std::vector<std::int64_t> after_transpose;
    // before_transpose and after_transpose are used to
    // prepare the "perm" argument of popart::transopose().
    for (int64_t i = 0; i < dimensions; ++i) {
      if (i < dim) {
        before_transpose.push_back(i);
        after_transpose.push_back(i);
      } else if (i > dim) {
        before_transpose.push_back(i);
        after_transpose.push_back(i - 1);
      } else {
        // i == dim
        after_transpose.push_back(dimensions - 1);
      }
    }
    before_transpose.push_back(dim);

    torch::jit::Node *transpose_before =
        createTranspose(graph, {node->input(0)}, before_transpose);

    torch::jit::Node *softmax =
        createSoftmax(graph, {transpose_before->output()}, dimensions - 1);
    // After moving the target dimension to the last dimension,
    // always do softmax on the last dimension: dimensions - 1.

    return createTranspose(graph, {softmax->output()}, after_transpose);

  } else {
    return createSoftmax(graph, {node->input(0)}, dim);
    // If "dim" is last dimension already, directly call popart::softmax()
  }
}

torch::jit::Node *logSoftmaxHandler(torch::jit::Graph *graph,
                                    torch::jit::Node *node) {
  // "aten::log_softmax(Tensor self, int dim, int? dtype) -> Tensor"

  torch::jit::Node *softmax = softmaxHandler(graph, node);
  // call softmaxHandler() to fix input dimension size > 2

  return createLog(graph, {softmax->output()});
}
} // namespace

// clang-format off
static bool handlers =
    registerHandlers(
        c10::aten::softmax, softmaxHandler,
        c10::aten::log_softmax, logSoftmaxHandler);
// clang-format on

} // namespace poptorch
