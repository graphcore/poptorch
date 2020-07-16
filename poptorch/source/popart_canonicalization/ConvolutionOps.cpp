// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch_logging/Logging.hpp"
#include <poptorch/OpBuilder.hpp>
#include <poptorch_logging/Error.hpp>

namespace poptorch {
namespace {
torch::jit::Node *convolutionHandler(torch::jit::Graph *graph,
                                     torch::jit::Node *node) {
  // aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[]
  //                    stride, int[] padding, int[] dilation, bool transposed,
  //                    int[] output_padding, int groups) -> Tensor
  std::optional<std::int64_t> transposed =
      handleConstant<std::int64_t>(node->input(6)->node());

  torch::jit::Value *input = node->input(0);
  torch::jit::Value *kernel = node->input(1);

  std::vector<torch::jit::Value *> inputs{input, kernel};

  if (!isNone(node->input(2)->node())) {
    inputs.push_back(node->input(2));
  }

  std::vector<std::int64_t> stride =
      handleList<int64_t>(node->input(3)->node());
  std::vector<std::int64_t> padding =
      handleList<std::int64_t>(node->input(4)->node());

  // Pytorch gives the padding as being the amount to pad in both
  // directions. Popart two arguments for each axis, the amount to pad in
  // each direction along that axis. In the form (Axis0Left, AxisNLeft...,
  // Axis0Right, AxisNRight) where left and right refer to the direction
  // along the axis to add zeros to.
  const std::size_t num_pads = padding.size();
  for (std::size_t pad_index = 0; pad_index < num_pads; ++pad_index) {
    padding.push_back(padding[pad_index]);
  }

  std::vector<std::int64_t> dilation =
      handleList<std::int64_t>(node->input(5)->node());
  // torch::jit::Value* output_padding = node->input(8);
  std::int64_t groups = *handleConstant<std::int64_t>(node->input(8)->node());

  if (transposed && *transposed == 0) {
    // Create a "normal" convolution.
    return poptorch::createConv(graph, inputs, dilation, groups, {}, padding,
                                stride);
  }
  ERROR("Transposed convolutions are not currently supported.");

  /* TODO(T22979) Re-enable once PopART supports transposed convolutions.
  // Output shape.
  // Give popart the shape of the output so it can autogenerate pads.
  std::vector<std::int64_t> outputShape = shapeFromTensor(node->output())

  return poptorch::createConvtranspose(graph, inputs, dilation,
    groups, {}, {}, outputShape, padding, stride);
  */
}

torch::jit::Node *conv2dHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride,
  //              int[] padding, int[] dilation, int groups) -> Tensor
  auto input = node->input(0);
  auto kernel = node->input(1);

  std::vector<torch::jit::Value *> inputs{input, kernel};

  // Add bias if present.
  if (!isNone(node->input(2)->node())) {
    inputs.push_back(node->input(2));
  }

  std::vector<std::int64_t> stride =
      handleList<std::int64_t>(node->input(3)->node());
  std::vector<std::int64_t> padding =
      handleList<std::int64_t>(node->input(4)->node());

  // Pytorch gives the padding as being the amount to pad in both
  // directions. Popart two arguments for each axis, the amount to pad in
  // each direction along that axis. In the form (Axis0Left, AxisNLeft...,
  // Axis0Right, AxisNRight) where left and right refer to the direction
  // along the axis to add zeros to.
  const std::size_t num_pads = padding.size();
  for (std::size_t pad_index = 0; pad_index < num_pads; ++pad_index) {
    padding.push_back(padding[pad_index]);
  }

  std::vector<std::int64_t> dilation =
      handleList<std::int64_t>(node->input(5)->node());
  std::int64_t groups = *handleConstant<std::int64_t>(node->input(6)->node());

  return poptorch::createConv(graph, inputs, dilation, groups, {}, padding,
                              stride);
}
} // namespace

static bool handlers =
    registerHandlers(c10::aten::_convolution, convolutionHandler,
                     c10::aten::conv2d, conv2dHandler);

} // namespace poptorch
