// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <mlir/IR/BuiltinTypes.h>

#include "dialect/Poptorch.hpp"
#include "lower_to_poplar/CompilerHelpers.hpp"

#include <poplar/ArrayRef.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/OperationDef.hpp>
#include <popops/Reduce.hpp>
#include <poputil/Broadcast.hpp>

namespace poptorch_ir {

// Helper function for cansting and performing reductions. The tensor in is cast
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
  auto out = in.elementType() != op_type
                 ? popops::cast(context.graph, in, op_type, context.seq)
                 : in;

  out = out.shape().empty()
            ? (op == popops::Operation::SQUARE_ADD
                   ? popops::square(context.graph, out, context.seq)
                   : out)
            : popops::reduce(context.graph, out, reduction_dims, {op},
                             context.seq);

  out = out.elementType() != out_type
            ? popops::cast(context.graph, out, out_type, context.seq)
            : out;

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

  context.tensors.insert({this->result(), output_tensor});
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

  context.tensors.insert({this->result(), var_output.variance});
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

  context.tensors.insert({this->result(), var_output.variance});
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

  context.tensors.try_emplace(this->result(), std::move(var_output.variance));
  context.tensors.try_emplace(this->mean(), std::move(var_output.mean));
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

  context.tensors.try_emplace(this->result(), std::move(var_output.variance));
  context.tensors.try_emplace(this->mean(), std::move(var_output.mean));
}

void reducesum::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  std::vector<std::size_t> dims = convertIntArray<std::size_t>(this->axes());

  auto output_tensor = reduceWithCasts(context, dims, in, out_type,
                                       popops::Operation::ADD, out_type);

  context.tensors.insert({this->result(), output_tensor});
}

void prod::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  std::vector<std::size_t> dims(in.shape().size(), 0);
  std::iota(dims.begin(), dims.end(), 0);

  auto output_tensor = reduceWithCasts(context, dims, in, out_type,
                                       popops::Operation::MUL, out_type);

  context.tensors.insert({this->result(), output_tensor});
}

void prod_dim::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  std::vector<std::size_t> dims{this->dim()};

  auto output_tensor = reduceWithCasts(context, dims, in, out_type,
                                       popops::Operation::MUL, out_type);

  context.tensors.insert({this->result(), output_tensor});
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

  context.tensors.insert({this->result(), output_tensor});
}

void all_out::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  std::vector<std::size_t> dims{this->dim()};

  auto output_tensor =
      reduceWithCasts(context, dims, in, poplar::BOOL,
                      popops::Operation::LOGICAL_AND, out_type);

  context.tensors.insert({this->result(), output_tensor});
}

void any::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  std::vector<std::size_t> dims(in.shape().size(), 0);
  std::iota(dims.begin(), dims.end(), 0);

  auto output_tensor = reduceWithCasts(context, dims, in, poplar::BOOL,
                                       popops::Operation::LOGICAL_OR, out_type);

  context.tensors.insert({this->result(), output_tensor});
}

void any_out::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  const auto out_type = CompilerContext::poplarTypeOf(
      this->result().getType().cast<mlir::RankedTensorType>().getElementType());

  std::vector<std::size_t> dims{this->dim()};

  auto output_tensor = reduceWithCasts(context, dims, in, poplar::BOOL,
                                       popops::Operation::LOGICAL_OR, out_type);

  context.tensors.insert({this->result(), output_tensor});
}

} // namespace poptorch_ir
