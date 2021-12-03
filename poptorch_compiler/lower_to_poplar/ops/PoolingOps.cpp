// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poplar/Graph.hpp>
#include <popnn/Pooling.hpp>

#include "lower_to_poplar/CompilerHelpers.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch_ir {

poplar::Tensor maxPool(CompilerContext &context, const poplar::Tensor &input,
                       const std::vector<std::size_t> &kernel_size,
                       const std::vector<unsigned> &stride,
                       const std::vector<std::size_t> &padding,
                       const std::vector<std::size_t> &dilation,
                       bool ceil_mode) {
  for (auto d : dilation) {
    ERROR_ON_MSG(
        d != 1,
        "Pooling dilations of value other than 1 are currently not supported.");
  }

  auto input_shape = input.shape();
  std::size_t n_spatial_dims = kernel_size.size();

  std::vector<std::size_t> input_size;
  // For each spatial dimension, PyTorch specifies only one padding value
  // for both the lower and upper sides of that dimension. Poplar takes lower
  // and upper padding for each spatial dimension
  std::vector<int> padding_lower;
  std::vector<int> padding_upper;
  for (auto s = 0u; s < n_spatial_dims; ++s) {
    // Poplar doesn't have a ceil mode. In order to reproduce the effect of
    // having one, we calculate how much extra (upper) padding we would need
    // to produce an output size one larger
    std::size_t extra_ceil_pad = 0;
    if (ceil_mode) {
      // Calculate the remainder from the division portion of the output size
      // calculation (from
      // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html)
      std::size_t remainder = (input_shape[2 + s] + 2 * padding[s] -
                               dilation[s] * (kernel_size[s] - 1) - 1) %
                              stride[s];
      // ceil_mode only makes a difference if ceil(x) != floor(x)
      if (remainder > 0) {
        extra_ceil_pad = stride[s] - remainder;
      }
    }
    padding_lower.push_back(padding[s]);
    padding_upper.push_back(padding[s] + extra_ceil_pad);
    input_size.push_back(input_shape[2 + s]);
  }

  popnn::pooling::PoolParams pool_params{
      popnn::PoolingType::MAX, input_size,    kernel_size,    stride,
      padding_lower,           padding_upper, input_shape[1], /* num_channels */
      input_shape[0],                                         /* batch_size */
      input.elementType()};

  return popnn::pooling::pool(context.graph, pool_params, input, context.seq);
}

void max_pool1d::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  auto kernel_size = convertIntArray<std::size_t>(this->kernel_size());
  auto stride = convertIntArray<unsigned>(this->stride());
  auto padding = convertIntArray<std::size_t>(this->padding());
  auto dilation = convertIntArray<std::size_t>(this->dilation());
  bool ceil_mode = this->ceil_mode();

  context.tensors[this->result()] = maxPool(context, input, kernel_size, stride,
                                            padding, dilation, ceil_mode);
}

void max_pool2d::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  auto kernel_size = convertIntArray<std::size_t>(this->kernel_size());
  auto stride = convertIntArray<unsigned>(this->stride());
  auto padding = convertIntArray<std::size_t>(this->padding());
  auto dilation = convertIntArray<std::size_t>(this->dilation());
  bool ceil_mode = this->ceil_mode();

  context.tensors[this->result()] = maxPool(context, input, kernel_size, stride,
                                            padding, dilation, ceil_mode);
}

void max_pool3d::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  auto kernel_size = convertIntArray<std::size_t>(this->kernel_size());
  auto stride = convertIntArray<unsigned>(this->stride());
  auto padding = convertIntArray<std::size_t>(this->padding());
  auto dilation = convertIntArray<std::size_t>(this->dilation());
  bool ceil_mode = this->ceil_mode();

  context.tensors[this->result()] = maxPool(context, input, kernel_size, stride,
                                            padding, dilation, ceil_mode);
}

// Implemented using average pooling until adaptive pooling is
// available in poplibs. This means the output will match PyTorch as
// long as the input dimensions are divisible by the output dimensions
poplar::Tensor adaptiveAvgPool(CompilerContext &context,
                               const poplar::Tensor &input,
                               const std::vector<std::size_t> &output_size) {
  std::size_t n_spatial_dims = output_size.size();
  std::vector<unsigned> stride(n_spatial_dims);
  std::vector<std::size_t> kernel_shape(n_spatial_dims);
  std::vector<std::size_t> input_shape = input.shape();
  std::vector<std::size_t> input_size;

  for (auto i = 0u; i < n_spatial_dims; i++) {
    std::size_t in_dim = input_shape[2 + i];
    input_size.push_back(in_dim);
    std::size_t out_dim = output_size[i];

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

  // We don't add padding in adaptive pooling
  std::vector<int> padding(n_spatial_dims, 0);

  popnn::pooling::PoolParams pool_params{popnn::PoolingType::AVG,
                                         input_size,
                                         kernel_shape,
                                         stride,
                                         padding,
                                         padding,
                                         input_shape[1], /* num_channels */
                                         input_shape[0], /* batch_size */
                                         input.elementType()};

  return popnn::pooling::pool(context.graph, pool_params, input, context.seq);
}

void adaptive_avg_pool1d::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());

  std::vector<std::size_t> output_size =
      convertIntArray<std::size_t>(this->output_size());
  context.tensors[this->result()] =
      adaptiveAvgPool(context, input, output_size);
}

void adaptive_avg_pool2d::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());

  std::vector<std::size_t> output_size =
      convertIntArray<std::size_t>(this->output_size());
  context.tensors[this->result()] =
      adaptiveAvgPool(context, input, output_size);
}

void adaptive_avg_pool3d::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());

  std::vector<std::size_t> output_size =
      convertIntArray<std::size_t>(this->output_size());
  context.tensors[this->result()] =
      adaptiveAvgPool(context, input, output_size);
}

} // namespace poptorch_ir
