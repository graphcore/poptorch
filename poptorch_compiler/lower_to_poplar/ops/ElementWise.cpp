// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <limits>

#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/ScaledAdd.hpp>

#include <popnn/NonLinearity.hpp>
#include <poptorch_logging/Error.hpp>

#include "../CompilerHelpers.hpp"
#include "dialect/Poptorch.hpp"

namespace pe = popops::expr;

namespace poptorch_ir {

#define BINARY_OP_RENAMED(name, popops_name)                                   \
  void name::lowerToPoplar(CompilerContext &context) {                         \
    poplar::Tensor input1 = context.fromSsa(this->in1());                      \
    poplar::Tensor input2 = context.fromSsa(this->in2());                      \
    poplar::Tensor out =                                                       \
        popops::popops_name(context.graph, input1, input2, context.seq);       \
    context.addTensor(this->result(), out);                                    \
  }

#define BINARY_OP(name) BINARY_OP_RENAMED(name, name)

#include "binary_ops.h.inc"

#undef BINARY_OP
#undef BINARY_OP_RENAMED

void add::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input1 = context.fromSsa(this->in1());
  poplar::Tensor const input2 = context.fromSsa(this->in2());

  const float value = this->alpha().convertToFloat();

  poplar::Tensor out;

  if (value != 1.0f) {
    auto expr = pe::Add(pe::_1, pe::Mul(pe::Const(value), pe::_2));
    out = popops::map(context.graph, expr, {input1, input2}, context.seq);
  } else {
    out = popops::add(context.graph, input1, input2, context.seq);
  }

  context.addTensor(this->result(), out);
}

void sub::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input1 = context.fromSsa(this->in1());
  poplar::Tensor const input2 = context.fromSsa(this->in2());

  const float value = this->alpha().convertToFloat();

  poplar::Tensor out;
  if (value != 1.0f) {
    auto expr = pe::Sub(pe::_1, pe::Mul(pe::Const(value), pe::_2));
    out = popops::map(context.graph, expr, {input1, input2}, context.seq);
  } else {
    out = popops::sub(context.graph, input1, input2, context.seq);
  }

  context.addTensor(this->result(), out);
}

void logicalXor::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input1 = context.fromSsa(this->in1());
  poplar::Tensor const input2 = context.fromSsa(this->in2());

  // Implicit casting is added to the input
  assert(input1.elementType() == poplar::BOOL);
  assert(input2.elementType() == poplar::BOOL);

  auto out = popops::map(context.graph, pe::BinaryOpType::NOT_EQUAL, input1,
                         input2, context.seq);

  context.addTensor(this->result(), out);
}

void floor_divide::lowerToPoplar(poptorch_ir::CompilerContext &context) {
  poplar::Tensor const input1 = context.fromSsa(this->in1());
  poplar::Tensor const input2 = context.fromSsa(this->in2());

  assert(input1.elementType().isFloatingPoint() ==
         input2.elementType().isFloatingPoint());

  if (input1.elementType().isFloatingPoint()) {
    auto out = popops::map(context.graph, pe::Trunc(pe::_1 / pe::_2),
                           {input1, input2}, context.seq);
    context.addTensor(this->result(), out);
  } else {
    auto out = popops::div(context.graph, input1, input2, context.seq);
    context.addTensor(this->result(), out);
  }
}

void remainder::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input1 = context.fromSsa(this->in1());
  poplar::Tensor const input2 = context.fromSsa(this->in2());

  // Compute the remainder rounding towards zero and correct it if the sign of
  // the remainder is negative
  auto r = pe::Rem(pe::_1, pe::_2);
  auto zero = pe::Const(0);
  auto op =
      pe::Select(r + pe::_2, r, (r != zero) && ((r < zero) != (pe::_2 < zero)));

  poplar::Tensor const out =
      popops::map(context.graph, op, {input1, input2}, context.seq);
  context.addTensor(this->result(), out);
}

#define UNARY_OP(name)                                                         \
  void name::lowerToPoplar(CompilerContext &context) {                         \
    poplar::Tensor input1 = context.fromSsa(this->in1());                      \
    poplar::Tensor out = popops::name(context.graph, input1, context.seq);     \
    context.addTensor(this->result(), out);                                    \
  }

#include "unary_ops.h.inc"

#undef UNARY_OP

void trunc::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const in1 = context.fromSsa(this->in1());

  auto out = popops::map(context.graph, popops::expr::UnaryOpType::TRUNC, in1,
                         context.seq);
  context.addTensor(this->result(), out);
}

void isnan::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const self = context.fromSsa(this->self());

  auto out = popops::map(context.graph, popops::expr::UnaryOpType::IS_NAN, self,
                         context.seq);
  context.addTensor(this->result(), out);
}

void addcmul::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input = context.fromSsa(this->input());
  poplar::Tensor const tensor1 = context.fromSsa(this->tensor1());
  poplar::Tensor const tensor2 = context.fromSsa(this->tensor2());
  const float value = this->value().convertToFloat();

  auto expr = pe::Add(pe::_1, pe::Mul(pe::_3, pe::_2));
  if (value != 1.0f) {
    expr = pe::Add(pe::_1, pe::Mul(pe::Const(value), pe::Mul(pe::_3, pe::_2)));
  }

  poplar::Tensor const out =
      popops::map(context.graph, expr, {input, tensor1, tensor2}, context.seq);

  context.addTensor(this->result(), out);
}

void addcdiv::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const input = context.fromSsa(this->input());
  poplar::Tensor const tensor1 = context.fromSsa(this->tensor1());
  poplar::Tensor const tensor2 = context.fromSsa(this->tensor2());
  const float value = this->value().convertToFloat();

  auto expr = pe::Add(pe::_1, pe::Divide(pe::_2, pe::_3));
  if (value != 1.0f) {
    expr =
        pe::Add(pe::_1, pe::Mul(pe::Const(value), pe::Divide(pe::_2, pe::_3)));
  }
  poplar::Tensor const out =
      popops::map(context.graph, expr, {input, tensor1, tensor2}, context.seq);

  context.addTensor(this->result(), out);
}

void clamp::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const self = context.fromSsa(this->self());

  auto min_optional = this->min();
  auto max_optional = this->max();

  const float min = min_optional.hasValue()
                        ? this->min().getValue().convertToDouble()
                        : std::numeric_limits<float>::lowest();

  const float max = max_optional.hasValue()
                        ? this->max().getValue().convertToDouble()
                        : std::numeric_limits<float>::max();

  // Note: pe::Clamp does not always behave in the correct manner when the min
  // value is more than the max value (it seems to happen on the ipu when we are
  // broadcasting)
  auto expr = pe::Min(pe::Max(pe::_1, pe::Const(min)), pe::Const(max));
  poplar::Tensor const out =
      popops::map(context.graph, expr, {self}, context.seq);

  context.addTensor(this->result(), out);
}

void clampTensor::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const self = context.fromSsa(this->self());
  poplar::Tensor min;
  poplar::Tensor max;
  const std::vector<uint64_t> param_shape = {self.shape()[0]};
  // creating numeric limit vectors

  min = this->min() ? context.fromSsa(this->min())
                    : createConstant(context, poplar::FLOAT, param_shape,
                                     std::numeric_limits<float>::lowest());
  max = this->max() ? context.fromSsa(this->max())
                    : createConstant(context, poplar::FLOAT, param_shape,
                                     std::numeric_limits<float>::max());

  // Note: pe::Clamp does not always behave in the correct manner when the min
  // value is more than the max value (it seems to happen on the ipu when we are
  // broadcasting)
  auto expr = pe::Min(pe::Max(pe::_1, pe::_2), pe::_3);

  poplar::Tensor const out =
      popops::map(context.graph, expr, {self, min, max}, context.seq);

  context.addTensor(this->result(), out);
}

void sigmoid_backward::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const grad_output = context.fromSsa(this->grad_output());
  poplar::Tensor const output = context.fromSsa(this->output());

  poplar::Tensor const out = popnn::nonLinearityInputGradient(
      context.graph, popnn::NonLinearityType::SIGMOID, output, grad_output,
      context.seq);
  context.addTensor(this->grad_input(), out);
}

void tanh_backward::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const grad_output = context.fromSsa(this->grad_output());
  poplar::Tensor const output = context.fromSsa(this->output());

  poplar::Tensor const out = popnn::nonLinearityInputGradient(
      context.graph, popnn::NonLinearityType::TANH, output, grad_output,
      context.seq);
  context.addTensor(this->grad_input(), out);
}

void threshold_out::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const self = context.fromSsa(this->self());
  const float threshold = this->threshold().convertToFloat();
  const float value = this->value().convertToFloat();

  // PyTorch treats NaN as greater than x, except when x = NaN or x = inf.
  const poplar::Tensor out =
      popops::map(context.graph,
                  pe::Select(pe::Const(value), pe::_1,
                             pe::Lte(pe::_1, pe::Const(threshold))),
                  {self}, context.seq);
  context.addTensor(this->result(), out);
}

void pow_Tensor_Scalar_out::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor const lhs = context.fromSsa(this->lhs());
  const float rhs = this->rhs().convertToFloat();
  poplar::Tensor const out = popops::map(
      context.graph, pe::Pow(pe::_1, pe::Const(rhs)), {lhs}, context.seq);
  context.addTensor(this->result(), out);
}

#define BINARY_CONDITIONAL_SCALAR_OP_IMPL(op, expr)                            \
  void op::lowerToPoplar(CompilerContext &context) {                           \
    const poplar::Tensor lhs = context.fromSsa(this->lhs());                   \
    const float rhs = this->rhs().convertToFloat();                            \
    const poplar::Tensor out = popops::map(                                    \
        context.graph, pe::expr(pe::_1, pe::Const(rhs)), {lhs}, context.seq);  \
    context.addTensor(this->result(), out);                                    \
  }

BINARY_CONDITIONAL_SCALAR_OP_IMPL(lt_Scalar_out, Lt)
BINARY_CONDITIONAL_SCALAR_OP_IMPL(le_Scalar_out, Lte)
BINARY_CONDITIONAL_SCALAR_OP_IMPL(gt_Scalar_out, Gt)
BINARY_CONDITIONAL_SCALAR_OP_IMPL(ge_Scalar_out, Gte)
BINARY_CONDITIONAL_SCALAR_OP_IMPL(eq_Scalar_out, Equal)
BINARY_CONDITIONAL_SCALAR_OP_IMPL(ne_Scalar_out, NotEqual)

#undef BINARY_CONDITIONAL_SCALAR_OP_IMPL
#define UNARY_OP_IMPL_EXPR(op, expr)                                           \
  void op::lowerToPoplar(CompilerContext &context) {                           \
    const poplar::Tensor input1 = context.fromSsa(this->in1());                \
    const poplar::Tensor out =                                                 \
        popops::map(context.graph, expr, {input1}, context.seq);               \
    context.addTensor(this->result(), out);                                    \
  }

UNARY_OP_IMPL_EXPR(acos,
                   pe::Const(static_cast<float>(M_PI_2)) - pe::Asin(pe::_1))
UNARY_OP_IMPL_EXPR(acosh, pe::Log(pe::_1 + pe::Sqrt(pe::Square(pe::_1) -
                                                    pe::Const(1.0f))))
UNARY_OP_IMPL_EXPR(atan, pe::Atan2(pe::_1, pe::Const(1.0f)))
UNARY_OP_IMPL_EXPR(atanh, pe::Const(0.5f) * pe::Log((pe::Const(1.0f) + pe::_1) /
                                                    (pe::Const(1.0f) - pe::_1)))
UNARY_OP_IMPL_EXPR(asinh, pe::Log(pe::_1 + pe::Sqrt(pe::Square(pe::_1) +
                                                    pe::Const(1.0f))))
UNARY_OP_IMPL_EXPR(sinh,
                   pe::Const(0.5f) *
                       (pe::Exp(pe::_1) - (pe::Const(1.0f) / pe::Exp(pe::_1))))
UNARY_OP_IMPL_EXPR(cosh,
                   pe::Const(0.5f) *
                       (pe::Exp(pe::_1) + (pe::Const(1.0f) / pe::Exp(pe::_1))))
UNARY_OP_IMPL_EXPR(reciprocal, pe::Const(1.0f) / pe::_1)
UNARY_OP_IMPL_EXPR(erfc, pe::Const(1.0f) - pe::Erf(pe::_1))
UNARY_OP_IMPL_EXPR(log10, pe::Log(pe::_1) / pe::Log(pe::Const(10.0f)))
UNARY_OP_IMPL_EXPR(log2, pe::Log(pe::_1) / pe::Log(pe::Const(2.0f)))
UNARY_OP_IMPL_EXPR(frac, pe::_1 - pe::Trunc(pe::_1))

#undef UNARY_OP_IMPL_EXPR

} // namespace poptorch_ir
