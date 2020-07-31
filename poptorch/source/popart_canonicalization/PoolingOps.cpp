// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {
torch::jit::Node *poolingHandler(torch::jit::Graph *graph,
                                 torch::jit::Node *node) {
  torch::jit::Symbol kind = node->kind();
  // aten::max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[]
  // padding, int[] dilation, bool ceil_mode) -> Tensor
  //
  // aten::avg_pool2d(Tensor self, int[] kernel_size, int[] stride, int[]
  //                   padding, bool ceil_mode, bool count_include_pad,
  //                   int? divisor_override) -> Tensor
  std::vector<std::int64_t> kernel_size =
      handleList<std::int64_t>(node->input(1)->node());
  std::vector<std::int64_t> stride =
      handleList<std::int64_t>(node->input(2)->node());
  std::vector<std::int64_t> padding =
      handleList<std::int64_t>(node->input(3)->node());

  // Pytorch gives the padding as being the amount to pad in both
  // directions. Popart two arguments for each axis, the amount to pad in
  // each direction along that axis. In the form (Axis0Left, AxisNLeft...,
  // Axis0Right, AxisNRight) where left and right refer to the direction
  // along the axis to add zeros to.
  const std::size_t num_pads = padding.size();
  for (std::size_t pad_index = 0; pad_index < num_pads; ++pad_index) {
    padding.push_back(padding[pad_index]);
  }

  if (kind == c10::aten::max_pool1d || kind == c10::aten::max_pool2d ||
      kind == c10::aten::max_pool3d) {
    return createMaxpool(graph, {node->input(0)}, 1, kernel_size, 0, {},
                         padding, 0, stride);
  }
  // ceil_mode, countIncludePad, divisor_override are ignored for now due
  // to not being supported directly in popart.
  return createAveragepool(graph, {node->input(0)}, kernel_size, 0, 0, padding,
                           stride);
}

torch::jit::Node *adaptivePoolingHandler(torch::jit::Graph *graph,
                                         torch::jit::Node *node) {
  torch::jit::Symbol kind = node->kind();
  // aten::adaptive_avg_pool2d(Tensor self, int[] output_size) -> Tensor
  // aten::adaptive_max_pool2d(Tensor self, int[] output_size) -> Tensor
  std::vector<std::int64_t> output_shape =
      handleList<std::int64_t>(node->input(1)->node());

  c10::TensorTypePtr as_tensor =
      node->input(0)->type()->cast<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();
  std::vector<std::int64_t> input_shape{*dims[2], *dims[3]};

  // Need to clean this code up.
  // TODO(tbd)
  const std::vector<int64_t> &stride{input_shape[0] / output_shape[0],
                                     input_shape[1] / output_shape[1]};

  const std::vector<int64_t> &kernel_shape{
      input_shape[0] - (output_shape[0] - 1) * stride[0],
      input_shape[1] - (output_shape[1] - 1) * stride[1]};
  const std::vector<int64_t> &padding{0, 0, 0, 0};

  if (kind == c10::aten::adaptive_avg_pool2d) {
    return createAveragepool(graph, {node->input(0)}, kernel_shape, 0, 0,
                             padding, stride);
  }
  ERROR("Adaptive max pooling isn't currently supported.");
  /* // TODO(T22978) Fix the number of inputs in PopParse so this can
     return 2.
     // Supported by Onnx.

      return poptorch::createMaxpool(graph,
     {node->input(0)}, 2, kernel_shape, padding, 0, stride);*/
}

} // namespace

// clang-format off
static bool handlers = registerHandlers(
    c10::aten::max_pool1d, poolingHandler,
    c10::aten::avg_pool1d, poolingHandler,
    c10::aten::max_pool2d, poolingHandler,
    c10::aten::avg_pool2d, poolingHandler,
    c10::aten::max_pool3d, poolingHandler,
    c10::aten::avg_pool3d, poolingHandler,
    c10::aten::adaptive_avg_pool2d, adaptivePoolingHandler,
    c10::aten::adaptive_max_pool2d, adaptivePoolingHandler);
// clang-format on

} // namespace poptorch
