// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "../PoptorchStaticInit.hpp"
#include "PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"
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

  // aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2]
  // stride=[], int[2] padding=[0, 0], int[2] dilation=[1, 1], bool
  // ceil_mode=False) -> (Tensor, Tensor)

  torch::jit::Value *x = node->input(0);
  auto kernel_size = constantToLongVec(node->input(1)->node());
  auto stride = constantToLongVec(node->input(2)->node());
  auto padding = constantToLongVec(node->input(3)->node());
  auto shape = shapeFromTensor(x);
  bool reshape_after = false;

  // The torch input might be missing the batch dimension, so add one if
  // necessary
  // (C, *in) -> (1, C, *in)
  if (shape.size() != stride.size() + 2) {
    shape.push_back(1);
    // simple rotation to the right
    std::rotate(shape.rbegin(), shape.rbegin() + 1, shape.rend());
    x = createReshape(graph, x, shape)->output();
    reshape_after = true;
  }

  // If we reshape, the output shape will be (1, C, *out) but torch expects
  // (C, *out)
  auto maybe_reshape_output = [&](torch::jit::Node *output) {
    if (reshape_after) {
      return createReshape(graph, output->output(),
                           shapeFromTensor(node->output()));
    }
    return output;
  };

  // Pytorch gives the padding as being the amount to pad in both
  // directions. Popart two arguments for each axis, the amount to pad in
  // each direction along that axis. In the form (Axis0Left, AxisNLeft...,
  // Axis0Right, AxisNRight) where left and right refer to the direction
  // along the axis to add zeros to.
  const std::size_t num_pads = padding.size();
  for (std::size_t pad_index = 0; pad_index < num_pads; ++pad_index) {
    padding.push_back(padding[pad_index]);
  }

  const bool is_max_pool = kind == c10::aten::max_pool1d ||
                           kind == c10::aten::max_pool2d ||
                           kind == c10::aten::max_pool3d ||
                           kind == c10::aten::max_pool1d_with_indices ||
                           kind == c10::aten::max_pool2d_with_indices ||
                           kind == c10::aten::max_pool3d_with_indices;

  if (is_max_pool) {
    auto dilations = constantToLongVec(node->input(4)->node());
    auto ceil_mode = constantToLong(node->input(5)->node());

    auto *output = createMaxpool(graph, {x}, 1, kernel_size, ceil_mode,
                                 dilations, padding, 0, stride);
    return maybe_reshape_output(output);
  }

  // divisor_override is ignored for now due to not being supported directly in
  // popart.
  auto ceil_mode = constantToLong(node->input(4)->node());

  bool count_include_pad = constantToBool(node->input(5)->node());
  // count_include_pad isn't supported in PopART so we check and pad manually if
  // the average pool is supposed to include the padding in its average.
  if (count_include_pad) {
    x = createConstantPad(graph, x, padding, 0.f)->output();
    // Ensure that padding isn't added twice.
    padding = {};
  }

  // popart only supports float types for avgpool
  auto input_type = getNodeScalarType(x);

  if (input_type == c10::kFloat) {
    auto *output = createAveragepool(graph, {x}, kernel_size, ceil_mode, 0,
                                     padding, stride);
    return maybe_reshape_output(output);
  }

  // all other types require casting via float
  x = createCast(graph, x, c10::kFloat)->output();
  x = createAveragepool(graph, {x}, kernel_size, ceil_mode, 0, padding, stride)
          ->output();
  auto *output = createCast(graph, x, input_type);
  return maybe_reshape_output(output);
}

torch::jit::Node *adaptivePoolingHandler(torch::jit::Graph *graph,
                                         torch::jit::Node *node) {
  // aten::adaptive_avg_pool1d(Tensor self, int[] output_size) -> Tensor
  // aten::adaptive_avg_pool2d(Tensor self, int[] output_size) -> Tensor
  // aten::adaptive_avg_pool3d(Tensor self, int[] output_size) -> Tensor

  torch::jit::Value *x = node->input(0);
  std::vector<std::int64_t> output_shape =
      constantToLongVec(node->input(1)->node());
  std::size_t n_output_dims = output_shape.size();

  std::vector<std::int64_t> input_shape = shapeFromTensor(x);
  std::size_t input_offset = input_shape.size() - n_output_dims;

  std::vector<std::int64_t> stride(n_output_dims);
  std::vector<std::int64_t> kernel_shape(n_output_dims);
  for (std::size_t i = 0; i < n_output_dims; i++) {
    std::int64_t in_dim = input_shape[input_offset + i];
    std::int64_t out_dim = output_shape[i];
    // This matches PyTorch's implementation as long as each input dim is
    // divisible by the corresponding output dim. If this is not the case, the
    // shape will be correct but the output will differ.
    if (in_dim % out_dim != 0) {
      std::stringstream ss;
      ss << "Input dim " << i << " (" << in_dim
         << ") is not divisible by the "
            "corresponding output dim ("
         << out_dim
         << "). The results will differ "
            "numerically from PyTorch's implementation.";
      ERROR(ss.str());
    }
    stride[i] = in_dim / out_dim;
    kernel_shape[i] = in_dim - (out_dim - 1) * stride[i];
  }

  std::vector<std::int64_t> padding(n_output_dims * 2, 0);
  return createAveragepool(graph, {x}, kernel_shape, 0, 0, padding, stride);
}

} // namespace

__attribute__((constructor(HANDLER_INIT_PRIORITY))) static void registration() {
  registerHandler(c10::aten::max_pool1d, poolingHandler);
  registerHandler(c10::aten::avg_pool1d, poolingHandler);
  registerHandler(c10::aten::max_pool2d, poolingHandler);
  registerHandler(c10::aten::avg_pool2d, poolingHandler);
  registerHandler(c10::aten::max_pool3d, poolingHandler);
  registerHandler(c10::aten::avg_pool3d, poolingHandler);

  registerHandler(c10::aten::max_pool1d_with_indices, poolingHandler);
  registerHandler(c10::aten::max_pool2d_with_indices, poolingHandler);
  registerHandler(c10::aten::max_pool3d_with_indices, poolingHandler);

  registerHandler(c10::aten::adaptive_avg_pool1d, adaptivePoolingHandler);
  registerHandler(c10::aten::adaptive_avg_pool2d, adaptivePoolingHandler);
  registerHandler(c10::aten::adaptive_avg_pool3d, adaptivePoolingHandler);
}

} // namespace poptorch
