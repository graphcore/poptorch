// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "dialect/PoptorchDialect.hpp"

#include <popnn/LogSoftmax.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>

#include "../CompilerHelpers.hpp"

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
  context.addTensor(this->result(), out);
}

void relu::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  poplar::Tensor out = popnn::nonLinearity(
      context.graph, popnn::NonLinearityType::RELU, input1, context.seq);
  context.addTensor(this->result(), out);
}

void gelu::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  poplar::Tensor out = popnn::nonLinearity(
      context.graph, popnn::NonLinearityType::GELU, input1, context.seq);
  context.addTensor(this->result(), out);
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
  context.addTensor(this->result(), out);
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
  context.addTensor(this->result(), out);
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

  context.addTensor(this->result(), out);
}

void logsoftmax::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  const std::uint32_t axis = this->dim();
  // If the axis is not along the last dimension.
  if (axis + 1 != input.rank()) {
    input = input.dimShufflePartial({axis, input.rank() - 1},
                                    {input.rank() - 1, axis});
  }
  poplar::Tensor out = popnn::logSoftmax(context.graph, input, context.seq);
  // Transpose it back into the correct form.
  if (axis + 1 != input.rank()) {
    out = out.dimShufflePartial({axis, out.rank() - 1}, {out.rank() - 1, axis});
  }
  context.addTensor(this->result(), out);
}

poplar::Tensor coerceTo2D(const poplar::Tensor &t) {
  const auto in_shape = t.shape();
  auto k = in_shape.begin();
  std::advance(k, t.rank() - 1);
  auto n = std::accumulate(in_shape.begin(), k, std::size_t{1},
                           std::multiplies<std::size_t>());
  return t.reshape({n, in_shape.back()});
}

void logsoftmax_backward::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor grad_output = context.fromSsa(this->grad_output());
  poplar::Tensor input = context.fromSsa(this->output());
  const std::uint32_t axis = this->dim();
  if (axis + 1 != input.rank()) {
    grad_output = grad_output.dimShufflePartial({axis, grad_output.rank() - 1},
                                                {grad_output.rank() - 1, axis});
    input = input.dimShufflePartial({axis, input.rank() - 1},
                                    {input.rank() - 1, axis});
  }
  auto grad_shape = grad_output.shape();
  grad_output = coerceTo2D(grad_output);
  input = coerceTo2D(input);

  // sum_j (g_j)
  std::vector<size_t> red_dims(input.rank() - 1);
  std::iota(red_dims.begin(), red_dims.end(), 1);
  std::vector<size_t> up_ranked(input.rank(), 1);
  up_ranked[0] = input.dim(0);
  poplar::Tensor sum_g = popops::reduce(context.graph, grad_output, red_dims,
                                        {popops::Operation::ADD}, context.seq)
                             .reshape(up_ranked);
  // softmax(x) = exp(log_softmax(x))
  poplar::Tensor prob = popops::exp(context.graph, input, context.seq);
  // g_i - softmax(x_i) * sum_j (g_j)
  auto dv = popops::map(context.graph, pe::Sub(pe::_1, pe::Mul(pe::_2, pe::_3)),
                        {grad_output, prob, sum_g}, context.seq);
  dv = dv.reshape(grad_shape);
  if (axis + 1 != grad_shape.size()) {
    dv = dv.dimShufflePartial({axis, dv.rank() - 1}, {dv.rank() - 1, axis});
  }
  context.addTensor(this->result(), dv);
}

} // namespace poptorch_ir
