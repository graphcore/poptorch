// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <llvm/ADT/APInt.h>

#include <poplar/Graph.hpp>
#include <popnn/Pooling.hpp>
#include <popops/Pad.hpp>

#include "lower_to_poplar/CompilerHelpers.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch_ir {

template <typename RemainderFn>
poplar::Tensor pool(CompilerContext &context, const poplar::Tensor &input,
                    const std::vector<std::size_t> &kernel_size,
                    const std::vector<unsigned> &stride,
                    const std::vector<std::size_t> &padding, bool ceil_mode,
                    popnn::PoolingType pool_type, RemainderFn &&remainder_fn) {
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
      // https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html)
      std::size_t remainder = remainder_fn(s);
      // ceil_mode only makes a difference if ceil(x) != floor(x)
      if (remainder > 0) {
        extra_ceil_pad = stride[s] - remainder;
      }
    }
    padding_lower.push_back(padding[s]);
    padding_upper.push_back(padding[s] + extra_ceil_pad);
    input_size.push_back(input.shape()[2 + s]);
  }

  popnn::pooling::PoolParams pool_params{
      pool_type,          input_size,    kernel_size,    stride,
      padding_lower,      padding_upper, input_shape[1], /* num_channels */
      input_shape[0],                                    /* batch_size */
      input.elementType()};

  return popnn::pooling::pool(context.graph, pool_params, input, context.seq);
}

void avg_pool::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  auto kernel_size = convertIntArray<std::size_t>(this->kernel_size());
  auto stride = convertIntArray<unsigned>(this->stride());
  auto padding = convertIntArray<std::size_t>(this->padding());
  bool ceil_mode = this->ceil_mode();
  if (this->count_include_pad()) {
    std::vector<std::ptrdiff_t> pad_vec = {0, 0};
    std::copy(padding.begin(), padding.end(), std::back_inserter(pad_vec));
    input = popops::pad(context.graph, input, pad_vec, pad_vec);
    for (auto &dim : padding) {
      dim = 0;
    }
  }
  llvm::Optional<std::uint64_t> divisor_override = this->divisor_override();
  ERROR_ON_MSG(divisor_override.hasValue(),
               "divisor_override is not supported for average pooling.");

  auto remainder_fn = [&](std::size_t s) {
    return (input.shape()[2 + s] + 2 * padding[s] - kernel_size[s]) % stride[s];
  };
  context.tensors[this->result()] =
      pool(context, input, kernel_size, stride, padding, ceil_mode,
           popnn::PoolingType::AVG, remainder_fn);
}

void max_pool::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  auto kernel_size = convertIntArray<std::size_t>(this->kernel_size());
  auto stride = convertIntArray<unsigned>(this->stride());
  auto padding = convertIntArray<std::size_t>(this->padding());
  auto dilation = convertIntArray<std::size_t>(this->dilation());
  bool ceil_mode = this->ceil_mode();

  for (auto d : dilation) {
    ERROR_ON_MSG(
        d != 1,
        "Pooling dilations of value other than 1 are currently not supported.");
  }

  auto remainder_fn = [&](std::size_t s) {
    return (input.shape()[2 + s] + 2 * padding[s] -
            dilation[s] * (kernel_size[s] - 1) - 1) %
           stride[s];
  };
  context.tensors[this->result()] =
      pool(context, input, kernel_size, stride, padding, ceil_mode,
           popnn::PoolingType::MAX, remainder_fn);
}

// Implemented using average pooling until adaptive pooling is
// available in poplibs. This means the output will match PyTorch as
// long as the input dimensions are divisible by the output dimensions
void adaptive_avg_pool::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  std::vector<std::size_t> output_size =
      convertIntArray<std::size_t>(this->output_size());
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

  context.tensors[this->result()] =
      popnn::pooling::pool(context.graph, pool_params, input, context.seq);
}

} // namespace poptorch_ir
