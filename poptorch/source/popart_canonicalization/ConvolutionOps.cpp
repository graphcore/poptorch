// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
namespace {

torch::jit::Node *convolutionHandler(torch::jit::Graph *graph,
                                     torch::jit::Node *node) {
  // aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[]
  //                    stride, int[] padding, int[] dilation, bool transposed,
  //                    int[] output_padding, int groups) -> Tensor
  bool transposed = constantToBool(node->input(6)->node());

  torch::jit::Value *input = node->input(0);
  torch::jit::Value *kernel = node->input(1);
  torch::jit::Value *bias = node->input(2);

  castWeightAndBias(graph, input, kernel, bias);

  std::vector<torch::jit::Value *> inputs{input, kernel};

  if (!isNone(bias->node())) {
    inputs.push_back(bias);
  }

  std::vector<std::int64_t> stride = constantToLongVec(node->input(3)->node());

  std::vector<std::int64_t> padding = constantToLongVec(node->input(4)->node());

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
      constantToLongVec(node->input(5)->node());

  std::vector<std::int64_t> output_padding =
      constantToLongVec(node->input(7)->node());

  std::int64_t groups = constantToLong(node->input(8)->node());

  if (!transposed) {
    // Create a "normal" convolution.

    // output_padding should be zero except for conv transpose
    for (auto out_pad : output_padding) {
      ERROR_ON(out_pad > 0);
    }

    return createConv(graph, inputs, dilation, groups, {}, padding, stride);
  }

  return createConvtranspose(graph, inputs, dilation, groups, {},
                             output_padding, {}, padding, stride);
}

torch::jit::Node *conv2dHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  // aten::conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride,
  //              int[] padding, int[] dilation, int groups) -> Tensor

  // Or:

  // aten::mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[]
  // padding, int[] stride, int[] dilation, int groups) -> (Tensor)

  torch::jit::Value *input = node->input(0);
  torch::jit::Value *kernel = node->input(1);
  torch::jit::Value *bias = node->input(2);

  castWeightAndBias(graph, input, kernel, bias);

  std::vector<torch::jit::Value *> inputs{input, kernel};

  if (!isNone(bias->node())) {
    inputs.push_back(bias);
  }

  const bool is_mkldnn_conv = node->kind() == c10::aten::mkldnn_convolution;
  const std::uint32_t stride_index = is_mkldnn_conv ? 4 : 3;
  const std::uint32_t padding_index = is_mkldnn_conv ? 3 : 4;

  std::vector<std::int64_t> stride =
      constantToLongVec(node->input(stride_index)->node());
  std::vector<std::int64_t> padding =
      constantToLongVec(node->input(padding_index)->node());

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
      constantToLongVec(node->input(5)->node());
  std::int64_t groups = constantToLong(node->input(6)->node());

  return poptorch::createConv(graph, inputs, dilation, groups, {}, padding,
                              stride);
}

torch::jit::Node *cumsumHandler(torch::jit::Graph *graph,
                                torch::jit::Node *node) {
  torch::jit::Value *data = node->input(0);
  std::vector<int64_t> data_shape = shapeFromTensor(data);
  int64_t dim = constantToLong(node->input(1)->node());
  int64_t r = static_cast<int64_t>(data_shape.size());
  ERROR_ON_MSG(dim < -r || dim > r - 1, "Dimension out of range.");

  if (dim < 0) {
    dim += r;
  }

  // By default, the output's `dtype` should match the input's.
  at::ScalarType requested_output_dtype = getNodeScalarType(data);

  if (node->inputs().size() == 4) {
    // We've been called with the form `torch.cumsum(..., out=output)`, so the
    // output tensor's `dtype` gets used as per the `torch.cumsum` spec.
    requested_output_dtype = getNodeScalarType(node->input(3));
  } else if (!isNone(node->input(2))) {
    // We've been called with an explicit `dtype`, so use that.
    requested_output_dtype = constantToScalarType(node->input(2)->node());
  }

  // We have to cast the input tensor to the output `dtype` *before* doing the
  // sum, to conform with the API of `torch.cumsum`.
  data = createCast(graph, data, requested_output_dtype)->output();

  // The 1-D conv kernel span is the size in the dim we are reducing along
  int64_t span = data_shape[static_cast<size_t>(dim)];

  if (span < 2) {
    // cumsum in singleton dimension or scalar/empty
    return createIdentity(graph, {data});
  }

  // Create the 1-d conv kernel
  std::vector<double> kernel_data(static_cast<size_t>(span), 1.0);
  torch::jit::Value *ones =
      createConstantFloatLike(graph, data, kernel_data, {span})->output();

  // ONNX conv expects the kernel to have size M x C/group X kW X kW
  // So reshape the kernel to have size [1,1,span,1]
  std::vector<int64_t> kernel_shape(4, 1);
  kernel_shape[2] = span;
  torch::jit::Value *k = createReshape(graph, ones, kernel_shape)->output();

  if (dim != 0) {
    // Transpose input so that we can apply the 1-d conv assuming dim==0
    std::vector<int64_t> p(r);
    std::iota(p.begin(), p.end(), 0);
    std::swap(p[0], p[dim]);
    data = createTranspose(graph, {data}, p)->output();
    std::swap(data_shape[0], data_shape[dim]);
  }

  // Coerce into [N,M] 2-d tensor
  if (r < 2) {
    data = createUnsqueeze(graph, {data}, {1})->output();
  }
  if (r > 2) {
    data = createFlatten(graph, {data}, 1)->output();
  }

  // ONNX conv expects the input data to have size batch X channel x H X W
  // So we reshape the [N,M] 2-d data to [M,1,N,1] and apply the 1-d conv
  // kernel of ones with [span-1,0] padding above and below.
  torch::jit::Value *x = createUnsqueeze(graph, {data}, {2, 3})->output();
  x = createTranspose(graph, {x}, {1, 2, 0, 3})->output();
  x = createCast(graph, x, c10::ScalarType::Float)->output();

  torch::jit::Value *y =
      createConv(graph, {x, k}, {}, 1, {}, {span - 1, 0, 0, 0}, {})->output();

  // Unfortunately we have to cast again here, because `createConv` always
  // returns a float-typed tensor. We can't *only* cast here either, because
  // cast -> sum != sum -> cast when going from float to int, and the spec of
  // `torch.cumsum` says to cast first. If we don't cast at all, info about our
  // size doesn't get transmitted to later ops relying on us (eg. `select`).
  y = createCast(graph, y, requested_output_dtype)->output();

  // Work back to the correct expected output shape
  y = createTranspose(graph, {y}, {2, 0, 1, 3})->output();
  y = createReshape(graph, y, data_shape)->output();

  if (dim != 0) {
    // Transpose back to the original axes orientation.
    std::vector<int64_t> p(r);
    std::iota(p.begin(), p.end(), 0);
    std::swap(p[0], p[dim]);
    y = createTranspose(graph, {y}, p)->output();
  }

  return y->node();
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::convolution, convolutionHandler);
  registerHandler(c10::aten::convolution_overrideable, convolutionHandler);
  registerHandler(c10::aten::_convolution, convolutionHandler);
  registerHandler(c10::aten::mkldnn_convolution, conv2dHandler);
  registerHandler(c10::aten::conv2d, conv2dHandler);
  registerHandler(c10::aten::cumsum, cumsumHandler);
}

} // namespace poptorch
