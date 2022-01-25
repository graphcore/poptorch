// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "dialect/PoptorchDialect.hpp"
#include "lower_to_poplar/CompilerHelpers.hpp"

#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>

namespace pe = popops::expr;

namespace poptorch_ir {

/*
 * Some 1:1 Pytorch -> Poplibs activation functions.
 */

// Outplace versions.
void swish::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  poplar::Tensor out = popnn::nonLinearity(
      context.graph, popnn::NonLinearityType::SWISH, input1, context.seq);
  context.tensors.insert({this->result(), out});
}

void relu::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  poplar::Tensor out = popnn::nonLinearity(
      context.graph, popnn::NonLinearityType::RELU, input1, context.seq);
  context.tensors.insert({this->result(), out});
}

void gelu::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  poplar::Tensor out = popnn::nonLinearity(
      context.graph, popnn::NonLinearityType::GELU, input1, context.seq);
  context.tensors.insert({this->result(), out});
}

// Inplace versions.
void hardsigmoid_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  // PyTorch formula is:
  // if x > 3 : return 1
  // if x < -3: return 0
  // return x / 6 + 0.5

  auto scaled_add =
      pe::Add(pe::Mul(pe::_1, pe::Const(1.0f / 6.0f)), pe::Const(0.5f));

  auto clamped = pe::Clamp(scaled_add, pe::Const(0.0f), pe::Const(1.0f));
  popops::mapInPlace(context.graph, clamped, {input1}, context.seq);
}

void hardswish_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  // PyTorch formula is:
  // if x > 3 : return x
  // if x < -3: return 0
  // return x * (x + 3) / 6

  // poly = x * (x + 3) / 6
  auto poly = pe::Mul(pe::Mul(pe::_1, pe::Add(pe::_1, pe::Const(3.0f))),
                      pe::Const(1.0f / 6.0f));

  auto less_than_neg3 = pe::Lt(pe::_1, pe::Const(-3.0f));
  auto more_than_three = pe::Gt(pe::_1, pe::Const(3.0f));

  // select( x < -3, 0.0, out)
  auto clamp_lower = pe::Select(pe::Const(0.0f), poly, less_than_neg3);
  // select( x > 3, 1.0, out)
  auto clamp_upper = pe::Select(pe::_1, clamp_lower, more_than_three);
  popops::mapInPlace(context.graph, clamp_upper, {input1}, context.seq);
}

void swish_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  popnn::nonLinearityInPlace(context.graph, popnn::NonLinearityType::SWISH,
                             input1, context.seq);
}

void relu_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  popnn::nonLinearityInPlace(context.graph, popnn::NonLinearityType::RELU,
                             input1, context.seq);
}

void gelu_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  popnn::nonLinearityInPlace(context.graph, popnn::NonLinearityType::GELU,
                             input1, context.seq);
}

void hardsigmoid::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->in1());
  // PyTorch formula is:
  // if x > 3 : return 1
  // if x < -3: return 0
  // return x / 6 + 0.5

  auto scaled_add =
      pe::Add(pe::Mul(pe::_1, pe::Const(1.0f / 6.0f)), pe::Const(0.5f));

  auto clamped = pe::Clamp(scaled_add, pe::Const(0.0f), pe::Const(1.0f));

  poplar::Tensor out =
      popops::map(context.graph, clamped, {input}, context.seq);
  context.tensors.insert({this->result(), out});
}

void hardswish::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->in1());
  // PyTorch formula is:
  // if x > 3 : return x
  // if x < -3: return 0
  // return x * (x + 3) / 6

  // poly = x * (x + 3) / 6
  auto poly = pe::Mul(pe::Mul(pe::_1, pe::Add(pe::_1, pe::Const(3.0f))),
                      pe::Const(1.0f / 6.0f));

  auto less_than_neg3 = pe::Lt(pe::_1, pe::Const(-3.0f));
  auto more_than_three = pe::Gt(pe::_1, pe::Const(3.0f));

  // select( x < -3, 0.0, out)
  auto clamp_lower = pe::Select(pe::Const(0.0f), poly, less_than_neg3);
  // select( x > 3, 1.0, out)
  auto clamp_upper = pe::Select(pe::_1, clamp_lower, more_than_three);

  poplar::Tensor out =
      popops::map(context.graph, clamp_upper, {input}, context.seq);
  context.tensors.insert({this->result(), out});
}

void softmax::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  const std::uint32_t axis = this->axis();

  // If the axis is not along the last dimension.
  if (axis + 1 != input.rank()) {
    input = input.dimShufflePartial({axis, input.rank() - 1},
                                    {input.rank() - 1, axis});
  }

  const std::vector<std::uint64_t> original_shape = input.shape();

  poplar::Tensor out = popnn::nonLinearity(
      context.graph, popnn::NonLinearityType::SOFTMAX_STABLE, input,
      context.seq);

  // Transpose it back into the correct form.
  if (axis + 1 != input.rank()) {
    out = out.dimShufflePartial({axis, out.rank() - 1}, {out.rank() - 1, axis});
  }

  context.tensors.insert({this->result(), out});
}

} // namespace poptorch_ir
