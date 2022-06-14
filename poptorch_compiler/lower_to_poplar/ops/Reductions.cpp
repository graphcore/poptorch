// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <mlir/IR/BuiltinTypes.h>

#include <algorithm>
#include <cstddef>
#include <functional>

#include "dialect/PoptorchDialect.hpp"

#include <poplar/ArrayRef.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <poplin/ConvParams.hpp>
#include <poplin/Convolution.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/OperationDef.hpp>
#include <popops/Pad.hpp>
#include <popops/Reduce.hpp>
#include <poputil/Broadcast.hpp>

#include "../CompilerHelpers.hpp"

namespace poptorch_ir {

inline poplar::Tensor mayCast(CompilerContext &context,
                              const poplar::Tensor &in,
                              const poplar::Type &cast_to) {
  return in.elementType() != cast_to
             ? popops::cast(context.graph, in, cast_to, context.seq)
             : in;
}

// Helper function for casting and performing reductions. The tensor in is cast
// to op_type, then the reduction op is applied and the output of that is cast
// to out_type. This function won't do casts that that aren't necessary
//
// Note: if the input tensor is a scalar tensor (i.e. the shape is empty) no
// reduction will be performed only casting will happen. This means that
// reduction_dims is ignored for scalar tensors
inline poplar::Tensor
reduceWithCasts(CompilerContext &context,
                const std::vector<std::size_t> &reduction_dims,
                const poplar::Tensor &in, const poplar::Type &op_type,
                popops::Operation op, const poplar::Type &out_type) {
  auto out = mayCast(context, in, op_type);

  out = out.shape().empty()
            ? (op == popops::Operation::SQUARE_ADD
                   ? popops::square(context.graph, out, context.seq)
                   : out)
            : popops::reduce(context.graph, out, reduction_dims, {op},
                             context.seq);

  out = mayCast(context, out, out_type);

  return out;
}

struct MeanAndVariance {
  poplar::Tensor mean;
  poplar::Tensor variance;
  // True if the number of elements that were reduced over is zero of one. In
  // this case the variance is either zeros or nans and the mean is either nans
  // or the original input
  bool is_trivial = false;
};

inline MeanAndVariance
computeMeanAndVariance(CompilerContext &context,
                       const std::vector<std::size_t> &dims,
                       const poplar::Tensor &in, bool useBesselCorrection) {
  // Note: although using the formula Var(X) = E[X^2] - (E[X])^2 is more
  // efficient computationally it has issues with numerical stability

  std::vector<std::size_t> mean_shape = in.shape();

  // If the input is a scalar just use a scalar here
  int64_t reduction_elts = 1;
  if (!mean_shape.empty()) {
    for (auto dim : dims) {
      reduction_elts *= mean_shape[dim];
      mean_shape[dim] = 1;
    }
  }

  if (reduction_elts == 0) {
    const auto mean_tensor =
        createConstant(context, in.elementType(), mean_shape,
                       std::numeric_limits<float>::quiet_NaN());
    const auto std_tensor =
        createConstant(context, in.elementType(), mean_shape,
                       std::numeric_limits<float>::quiet_NaN());
    return {mean_tensor, std_tensor, true};
  }
  if (reduction_elts == 1) {
    const auto std_tensor = createConstant(
        context, in.elementType(), mean_shape,
        useBesselCorrection ? std::numeric_limits<float>::quiet_NaN() : 0.0f);
    return {in, std_tensor, true};
  }
  if (in.numElements() == 0) {
    const auto std_tensor =
        createConstant(context, in.elementType(), mean_shape, 0.0f);
    return {in, std_tensor, true};
  }

  const auto mean_denominator =
      createConstant(context, in.elementType(), {}, 1.0f / reduction_elts);

  auto mean_tensor = popops::reduce(
      context.graph, in, dims,
      {popops::Operation::ADD, false, mean_denominator}, context.seq);

  auto broadcast_mean_tensor = mean_tensor.reshape(mean_shape);

  poputil::broadcastToMatch(broadcast_mean_tensor, in.shape());
  auto zero_mean_tensor =
      popops::sub(context.graph, in, broadcast_mean_tensor, context.seq);

  // Perform the inverse divide.
  const auto std_denominator = createConstant(
      context, in.elementType(), {},
      1.0f / (useBesselCorrection ? (reduction_elts - 1) : reduction_elts));

  auto output_tensor = popops::reduce(
      context.graph, zero_mean_tensor, dims,
      {popops::Operation::SQUARE_ADD, false, std_denominator}, context.seq);

  return {mean_tensor, output_tensor, false};
}

void reducemean::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  std::vector<std::size_t> dims = convertIntArray<std::size_t>(this->axes());

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  auto output_tensor = reduceWithCasts(context, dims, in, out_type,
                                       popops::Operation::ADD, out_type);

  // Perform the inverse divide.
  const int64_t in_elms =
      this->input().getType().cast<mlir::RankedTensorType>().getNumElements();
  const int64_t out_elms =
      this->result().getType().cast<mlir::RankedTensorType>().getNumElements();
  const float ratio =
      static_cast<float>(out_elms) / static_cast<float>(in_elms);
  popops::mulInPlace(context.graph, output_tensor, ratio, context.seq);

  // In case keep_dim = True
  output_tensor = reshapeToMLIRShape(output_tensor, this->result().getType());

  context.addTensor(this->result(), output_tensor);
}

void std_correction::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->self());

  const auto out_tensor_type =
      this->result().getType().cast<mlir::RankedTensorType>();
  const auto out_shape_llvm = out_tensor_type.getShape();
  const std::vector<std::size_t> out_shape{out_shape_llvm.begin(),
                                           out_shape_llvm.end()};

  std::vector<std::size_t> dims =
      convertIntArray<std::size_t>(this->dim().getValue());

  auto var_output = computeMeanAndVariance(context, dims, in,
                                           this->correction().getValue() != 0);

  // If the variance is zeros or nans we don't need to take a sqrt
  if (!var_output.is_trivial) {
    popops::sqrtInPlace(context.graph, var_output.variance, context.seq);
  }

  // In case keep_dim = True
  var_output.variance =
      reshapeToMLIRShape(var_output.variance, this->result().getType());

  context.addTensor(this->result(), var_output.variance);
}
void var_correction::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->self());

  const auto out_tensor_type =
      this->result().getType().cast<mlir::RankedTensorType>();
  const auto out_shape_llvm = out_tensor_type.getShape();
  const std::vector<std::size_t> out_shape{out_shape_llvm.begin(),
                                           out_shape_llvm.end()};

  std::vector<std::size_t> dims =
      convertIntArray<std::size_t>(this->dim().getValue());

  auto var_output = computeMeanAndVariance(context, dims, in,
                                           this->correction().getValue() != 0);

  // In case keep_dim = True
  var_output.variance =
      reshapeToMLIRShape(var_output.variance, this->result().getType());

  context.addTensor(this->result(), var_output.variance);
}
void std_mean_correction::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->self());

  const auto out_tensor_type =
      this->result().getType().cast<mlir::RankedTensorType>();
  const auto out_shape_llvm = out_tensor_type.getShape();
  const std::vector<std::size_t> out_shape{out_shape_llvm.begin(),
                                           out_shape_llvm.end()};

  std::vector<std::size_t> dims =
      convertIntArray<std::size_t>(this->dim().getValue());

  auto var_output = computeMeanAndVariance(context, dims, in,
                                           this->correction().getValue() != 0);

  // If the variance is zeros on nans we don't need to take a sqrt
  if (!var_output.is_trivial) {
    popops::sqrtInPlace(context.graph, var_output.variance, context.seq);
  }

  // In case keep_dim = True
  var_output.mean = reshapeToMLIRShape(var_output.mean, this->mean().getType());
  var_output.variance =
      reshapeToMLIRShape(var_output.variance, this->result().getType());

  context.addTensor(this->result(), var_output.variance);
  context.addTensor(this->mean(), var_output.mean);
}
void var_mean_correction::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->self());

  const auto out_tensor_type =
      this->result().getType().cast<mlir::RankedTensorType>();
  const auto out_shape_llvm = out_tensor_type.getShape();
  const std::vector<std::size_t> out_shape{out_shape_llvm.begin(),
                                           out_shape_llvm.end()};

  std::vector<std::size_t> dims =
      convertIntArray<std::size_t>(this->dim().getValue());

  auto var_output = computeMeanAndVariance(context, dims, in,
                                           this->correction().getValue() != 0);

  // In case keep_dim = True
  var_output.mean = reshapeToMLIRShape(var_output.mean, this->mean().getType());
  var_output.variance =
      reshapeToMLIRShape(var_output.variance, this->result().getType());

  context.addTensor(this->result(), var_output.variance);
  context.addTensor(this->mean(), var_output.mean);
}

void reducesum::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  std::vector<std::size_t> dims = convertIntArray<std::size_t>(this->axes());

  auto output_tensor = reduceWithCasts(context, dims, in, out_type,
                                       popops::Operation::ADD, out_type);

  // In case keep_dim = True
  output_tensor = reshapeToMLIRShape(output_tensor, this->result().getType());

  context.addTensor(this->result(), output_tensor);
}

void prod::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  std::vector<std::size_t> dims(in.shape().size(), 0);
  std::iota(dims.begin(), dims.end(), 0);

  auto output_tensor = reduceWithCasts(context, dims, in, out_type,
                                       popops::Operation::MUL, out_type);

  context.addTensor(this->result(), output_tensor);
}

void prod_dim::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  std::vector<std::size_t> dims{this->dim()};

  auto output_tensor = reduceWithCasts(context, dims, in, out_type,
                                       popops::Operation::MUL, out_type);

  // In case keep_dim = True
  output_tensor = reshapeToMLIRShape(output_tensor, this->result().getType());

  context.addTensor(this->result(), output_tensor);
}

void all::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  std::vector<std::size_t> dims(in.shape().size(), 0);
  std::iota(dims.begin(), dims.end(), 0);

  auto output_tensor =
      reduceWithCasts(context, dims, in, poplar::BOOL,
                      popops::Operation::LOGICAL_AND, out_type);

  context.addTensor(this->result(), output_tensor);
}

void all_out::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  std::vector<std::size_t> dims{this->dim()};

  auto output_tensor =
      reduceWithCasts(context, dims, in, poplar::BOOL,
                      popops::Operation::LOGICAL_AND, out_type);

  // In case keep_dim = True
  output_tensor = reshapeToMLIRShape(output_tensor, this->result().getType());

  context.addTensor(this->result(), output_tensor);
}

void any::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  std::vector<std::size_t> dims(in.shape().size(), 0);
  std::iota(dims.begin(), dims.end(), 0);

  auto output_tensor = reduceWithCasts(context, dims, in, poplar::BOOL,
                                       popops::Operation::LOGICAL_OR, out_type);

  context.addTensor(this->result(), output_tensor);
}

void any_out::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  std::vector<std::size_t> dims{this->dim()};

  auto output_tensor = reduceWithCasts(context, dims, in, poplar::BOOL,
                                       popops::Operation::LOGICAL_OR, out_type);

  // In case keep_dim = True
  output_tensor = reshapeToMLIRShape(output_tensor, this->result().getType());

  context.addTensor(this->result(), output_tensor);
}

// Perform a prefix sum by doing a convolution with an array of ones
//
// E.g. if we have a 1d array [1, 2, 3] we pad before to with N - 1 zeros to get
// [0, 0, 1, 2, 3] then perform a convolution with N ones.
//
// [0, 0, 1, 2, 3]
// [1, 1, 1]                   = 1
//    [1, 1, 1]    = 1 + 2     = 3
//       [1, 1, 1] = 1 + 2 + 3 = 6
//
// This gives the partial sum.
//
// Note: this isn't the quickest algorithm for performing a prefix sum on the
// ipu it is however easy to implement
void cumsum_out::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  const auto in_shape = in.shape();

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  // The pytorch api states we cast to the output_type before doing the partial
  // sum
  auto cast_in = mayCast(context, in, out_type);

  // We don't need to do anything if the input is a scalar or if there are no
  // elements in the input
  if (in.numElements() == 0 || in_shape.empty()) {
    context.addTensor(this->result(), cast_in);
    return;
  }

  const auto dim = this->dim();
  const auto sum_size = in_shape[dim];

  // We don't need to do anything if the the sum is over one element
  if (sum_size == 1) {
    context.addTensor(this->result(), cast_in);
    return;
  }

  // poplibs can only do convolutions on floating point types. If the type
  // isn't a half we cast to a float to attempt to avoid precision loss
  // Note: there may be two casts in a row here but we cannot removed either
  const auto working_type =
      out_type == poplar::HALF ? poplar::HALF : poplar::FLOAT;
  cast_in = mayCast(context, cast_in, working_type);

  // Convolution input is of the form [B x inChans x W] we will convolve
  // over W and set H and inChans to 1
  const auto swapped_to_last_channel = cast_in.dimRoll(dim, cast_in.rank() - 1);

  auto ignored_shape = swapped_to_last_channel.shape();
  ignored_shape.pop_back();

  const auto flattened =
      swapped_to_last_channel.flatten(0, swapped_to_last_channel.rank() - 1);
  const auto flattened_shape = flattened.shape();
  const auto ignored_dims = flattened_shape.front();

  const auto reshaped =
      flattened.reshape({ignored_dims, 1, flattened_shape.back()});

  const auto to_convolve = popops::pad(context.graph, reshaped, sum_size - 1, 0,
                                       reshaped.rank() - 1, 0.0f);

  // The weights tensor has shape
  // [convGroups x outChansPerConvGroup x inChansPerConvGroup x W]
  std::vector<std::size_t> ones_shape{1, 1, 1, sum_size};
  const auto ones =
      createConstant(context, to_convolve.elementType(), ones_shape, 1);

  poplin::ConvParams params(to_convolve.elementType() /*input type*/,
                            to_convolve.dim(0) /*batch_size*/,
                            {to_convolve.dim(2)} /*input field shape*/,
                            {ones_shape.back()} /*kernel shape*/,
                            1 /*input channels per group*/,
                            1 /*output channels per group*/, 1 /*groups*/);
  const auto convolved = poplin::convolution(context.graph, to_convolve, ones,
                                             params, false, context.seq);

  ERROR_ON(convolved.shape() !=
           std::vector<std::size_t>({ignored_dims, 1, sum_size}));

  const auto convolved_swapped_to_last_channel =
      convolved.reshapePartial(0, 2, ignored_shape);
  const auto prefix_sum = convolved_swapped_to_last_channel.dimRoll(
      convolved_swapped_to_last_channel.rank() - 1, dim);

  ERROR_ON(in_shape != prefix_sum.shape());

  // Since we cast to a floating point type to do the sum we must cast back to
  // the output type
  const auto output = mayCast(context, prefix_sum, out_type);

  context.addTensor(this->result(), output);
}

} // namespace poptorch_ir
