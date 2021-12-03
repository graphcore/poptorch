// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poplar/Graph.hpp>
#include <popnn/Pooling.hpp>

#include "lower_to_poplar/CompilerHelpers.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch_ir {

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
