// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "dialect/Poptorch.hpp"
#include "dialect/PoptorchDialect.hpp"

#include <poplar/Tensor.hpp>
#include <popnn/LogSoftmax.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>
#include <poprand/RandomGen.hpp>

#include "../CompilerHelpers.hpp"

namespace pe = popops::expr;

namespace poptorch_ir {

/*
 * Some 1:1 Pytorch -> Poplibs activation functions.
 */

// Outplace versions.
void swish::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input1 = context.fromSsa(this->in1());
  poplar::Tensor const out = popnn::nonLinearity(
      context.graph, popnn::NonLinearityType::SWISH, input1, context.seq);
  context.addTensor(this->result(), out);
}

void elu::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input1 = context.fromSsa(this->in1());
  const auto alpha = this->alpha().convertToFloat();
  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;

  exprs.push_back(std::make_unique<pe::Mul>(
      pe::Const(alpha), pe::Sub(pe::Exp(pe::_1), pe::Const(1.0f))));
  exprs.push_back(std::make_unique<pe::Min>(pe::Const(0.0f), *exprs.back()));
  exprs.push_back(std::make_unique<pe::Add>(pe::Max(pe::Const(0.0f), pe::_1),
                                            *exprs.back()));

  const poplar::Tensor output =
      popops::map(context.graph, *exprs.back(), {input1}, context.seq);
  context.addTensor(this->result(), output);
}

void hardshrink::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input1 = context.fromSsa(this->in1());
  const float lambd = this->lambd().convertToFloat();
  const poplar::Tensor output =
      popops::map(context.graph,
                  pe::Select(pe::Const(0.0f), pe::_1,
                             pe::And(pe::Gte(pe::_1, pe::Const(-lambd)),
                                     pe::Lte(pe::_1, pe::Const(lambd)))),
                  {input1}, context.seq);
  context.addTensor(this->result(), output);
}

void softshrink_out::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input1 = context.fromSsa(this->in1());
  const float lambd = this->lambd().convertToFloat();
  const poplar::Tensor output = popops::map(
      context.graph,
      pe::Select(pe::Add(pe::_1, pe::Const(lambd)),
                 pe::Select(pe::Sub(pe::_1, pe::Const(lambd)), pe::Const(0.0f),
                            pe::Gt(pe::_1, pe::Const(lambd))),
                 pe::Lt(pe::_1, pe::Const(-lambd))),
      {input1}, context.seq);
  context.addTensor(this->result(), output);
}

void rrelu_with_noise::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const self = context.fromSsa(this->self());
  const float lower = this->lower().convertToFloat();
  const float upper = this->upper().convertToFloat();

  emitWarning("`noise' argument to RReLU is currently ignored");

  poplar::Tensor const val =
      this->training()
          ? poprand::uniform(context.graph, nullptr, 0,
                             createConstant(context, poplar::FLOAT, {}, 0.0f),
                             poplar::FLOAT, upper, lower, context.seq)
          : createConstant(context, poplar::FLOAT, {}, (lower + upper) / 2);

  const poplar::Tensor output =
      popops::map(context.graph,
                  pe::Select(pe::Mul(pe::_1, pe::_2), pe::_1,
                             pe::Lt(pe::_1, pe::Const(0.0f))),
                  {self, val}, context.seq);
  context.addTensor(this->result(), output);
}

void prelu::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const self = context.fromSsa(this->self());
  poplar::Tensor const weight = context.fromSsa(this->weight());
  const poplar::Tensor output =
      popops::map(context.graph,
                  pe::Select(pe::Mul(pe::_1, pe::_2), pe::_1,
                             pe::Lt(pe::_1, pe::Const(0.0f))),
                  {self, weight}, context.seq);
  context.addTensor(this->result(), output);
}

void log_sigmoid_forward::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const in1 = context.fromSsa(this->in1());
  const poplar::Tensor output = popops::map(
      context.graph, pe::Log(pe::Sigmoid(pe::_1)), {in1}, context.seq);
  context.addTensor(this->output(), output);
}

void softplus::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input = context.fromSsa(this->input());
  const float beta = this->beta().convertToFloat();
  const float threshold = this->threshold().convertToFloat();

  std::vector<std::unique_ptr<pe::Expr>> exprs;
  exprs.push_back(std::make_unique<pe::PlaceHolder>(pe::_1));

  if (beta != 1.0f) {
    exprs.push_back(std::make_unique<pe::Mul>(pe::Const(beta), *exprs.back()));
  }

  // log1p(-exp(|beta * x|))
  exprs.push_back(std::make_unique<pe::Exp>(-pe::Abs(*exprs.back())));
  exprs.push_back(std::make_unique<pe::Log1p>(*exprs.back()));

  if (beta != 1.0f) {
    exprs.push_back(
        std::make_unique<pe::Divide>(*exprs.back(), pe::Const(beta)));
  }

  // 1/beta * log1p(-exp(|beta * x|)) + max(x, 0)
  exprs.push_back(std::make_unique<pe::Add>(*exprs.back(),
                                            pe::Max(pe::_1, pe::Const(0.0f))));

  // beta * x <= threshold ? 1/beta * log1p(-exp(|beta * x|)) + max(x, 0) : x
  exprs.push_back(std::make_unique<pe::Select>(
      *exprs.back(), pe::_1, *exprs.back() <= pe::Const(threshold)));

  const poplar::Tensor output =
      popops::map(context.graph, *exprs.back(), {input}, context.seq);
  context.addTensor(this->result(), output);
}

void relu::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input1 = context.fromSsa(this->in1());
  poplar::Tensor const out = popnn::nonLinearity(
      context.graph, popnn::NonLinearityType::RELU, input1, context.seq);
  context.addTensor(this->result(), out);
}

void leaky_relu::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input1 = context.fromSsa(this->in1());
  const float negative_slope = this->negative_slope().convertToFloat();

  auto expression = pe::Select(pe::Mul(pe::Const(negative_slope), pe::_1),
                               pe::_1, pe::Lt(pe::_1, pe::Const(0.0f)));

  poplar::Tensor const out =
      popops::map(context.graph, expression, {input1}, context.seq);
  context.addTensor(this->result(), out);
}

void leaky_relu_backward::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const grad_output = context.fromSsa(this->grad_output());
  poplar::Tensor const self = context.fromSsa(this->self());
  const float negative_slope = this->negative_slope().convertToFloat();
  const bool self_is_result = this->self_is_result();

  // This error message is normally raised by PyTorch at the
  // dispatch function level so we need to raise it ourselves
  ERROR_ON_MSG(self_is_result && negative_slope < 0.0,
               "PyTorch: In-place leakyReLu backward calculation is triggered "
               "with a negative slope which is not supported. "
               "This is caused by calling in-place forward function with a "
               "negative slope, "
               "please call out-of-place version instead. File an issue at "
               "https://github.com/pytorch/pytorch if you do "
               "require supporting in-place leakRelu backward calculation with "
               "negative slope");

  auto expression = pe::Select(pe::Mul(pe::Const(negative_slope), pe::_1),
                               pe::_1, pe::Lt(pe::_2, pe::Const(0.0f)));

  auto out =
      popops::map(context.graph, expression, {grad_output, self}, context.seq);
  context.addTensor(this->grad_input(), out);
}

void gelu::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input1 = context.fromSsa(this->in1());
  poplar::Tensor const out = popnn::nonLinearity(
      context.graph, popnn::NonLinearityType::GELU, input1, context.seq);
  context.addTensor(this->result(), out);
}

void hardsigmoid::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input = context.fromSsa(this->in1());
  // PyTorch formula is:
  // if x > 3 : return 1
  // if x < -3: return 0
  // return x / 6 + 0.5

  auto scaled_add =
      pe::Add(pe::Mul(pe::_1, pe::Const(1.0f / 6.0f)), pe::Const(0.5f));

  auto clamped = pe::Clamp(scaled_add, pe::Const(0.0f), pe::Const(1.0f));

  poplar::Tensor const out =
      popops::map(context.graph, clamped, {input}, context.seq);
  context.addTensor(this->result(), out);
}

void hardswish::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input = context.fromSsa(this->in1());
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

  poplar::Tensor const out =
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
  std::vector<std::size_t> red_dims(input.rank() - 1);
  std::iota(red_dims.begin(), red_dims.end(), 1);
  std::vector<std::size_t> up_ranked(input.rank(), 1);
  up_ranked[0] = input.dim(0);
  poplar::Tensor const sum_g =
      popops::reduce(context.graph, grad_output, red_dims,
                     {popops::Operation::ADD}, context.seq)
          .reshape(up_ranked);
  // softmax(x) = exp(log_softmax(x))
  poplar::Tensor const prob = popops::exp(context.graph, input, context.seq);
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
