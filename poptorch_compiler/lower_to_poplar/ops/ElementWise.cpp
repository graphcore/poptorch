// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>

#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/ScaledAdd.hpp>

#include <popnn/NonLinearity.hpp>

#include "../CompilerHelpers.hpp"


namespace pe = popops::expr;

namespace poptorch_ir {

#define BINARY_OP_OUTPLACE(name, popops_name)                                  \
  void name::lowerToPoplar(CompilerContext &context) {                         \
    poplar::Tensor input1 = context.fromSsa(this->in1());                      \
    poplar::Tensor input2 = context.fromSsa(this->in2());                      \
    poplar::Tensor out =                                                       \
        popops::popops_name(context.graph, input1, input2, context.seq);       \
    context.addTensor(this->result(), out);                                    \
  }
#define BINARY_OP_INPLACE(name, popops_name)                                   \
  void name##_::lowerToPoplar(CompilerContext &context) {                      \
    poplar::Tensor input1 = context.fromSsa(this->in1());                      \
    poplar::Tensor input2 = context.fromSsa(this->in2());                      \
    popops::popops_name##InPlace(context.graph, input1, input2, context.seq);  \
  }

#define BINARY_OP(name)                                                        \
  BINARY_OP_OUTPLACE(name, name)                                               \
  BINARY_OP_INPLACE(name, name)

#include "binary_ops.h.inc"

#undef BINARY_OP
#undef BINARY_OP_INPLACE
#undef BINARY_OP_OUTPLACE

void add::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  poplar::Tensor input2 = context.fromSsa(this->in2());

  float value = this->alpha().convertToFloat();

  poplar::Tensor out;

  if (value != 1.0f) {
    auto expr = pe::Add(pe::_1, pe::Mul(pe::Const(value), pe::_2));
    out = popops::map(context.graph, expr, {input1, input2}, context.seq);
  } else {
    out = popops::add(context.graph, input1, input2, context.seq);
  }

  context.addTensor(this->result(), out);
}

void add_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  poplar::Tensor input2 = context.fromSsa(this->in2());
  float value = this->alpha().convertToFloat();

  if (value != 1.0f) {
    if (input1.shape() == input2.shape()) {
      popops::scaledAddTo(context.graph, input1, input2, value, context.seq);
    } else {
      auto expr = pe::Add(pe::_1, pe::Mul(pe::Const(value), pe::_2));
      popops::mapInPlace(context.graph, expr, {input1, input2}, context.seq);
    }
  } else {
    popops::addInPlace(context.graph, input1, input2, context.seq);
  }
}

void sub::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  poplar::Tensor input2 = context.fromSsa(this->in2());

  float value = this->alpha().convertToFloat();

  poplar::Tensor out;
  if (value != 1.0f) {
    auto expr = pe::Sub(pe::_1, pe::Mul(pe::Const(value), pe::_2));
    out = popops::map(context.graph, expr, {input1, input2}, context.seq);
  } else {
    out = popops::sub(context.graph, input1, input2, context.seq);
  }

  context.addTensor(this->result(), out);
}

void sub_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  poplar::Tensor input2 = context.fromSsa(this->in2());
  float value = this->alpha().convertToFloat();

  if (value != 1.0f) {
    if (input1.shape() == input2.shape()) {
      popops::scaledSubtractFrom(context.graph, input1, input2, value,
                                 context.seq);
    } else {
      auto expr = pe::Sub(pe::_1, pe::Mul(pe::Const(value), pe::_2));
      popops::mapInPlace(context.graph, expr, {input1, input2}, context.seq);
    }
  } else {
    popops::subInPlace(context.graph, input1, input2, context.seq);
  }
}

#define UNARY_OP(name)                                                         \
  void name::lowerToPoplar(CompilerContext &context) {                         \
    poplar::Tensor input1 = context.fromSsa(this->in1());                      \
    poplar::Tensor out = popops::name(context.graph, input1, context.seq);     \
    context.addTensor(this->result(), out);                                    \
  }                                                                            \
  void name##_::lowerToPoplar(CompilerContext &context) {                      \
    poplar::Tensor input1 = context.fromSsa(this->in1());                      \
    popops::name##InPlace(context.graph, input1, context.seq);                 \
  }

#include "unary_ops.h.inc"

#undef UNARY_OP

void isnan::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor self = context.fromSsa(this->self());

  auto out = popops::map(context.graph, popops::expr::UnaryOpType::IS_NAN, self,
                         context.seq);
  context.addTensor(this->result(), out);
}

void scaledadd_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  poplar::Tensor input2 = context.fromSsa(this->in2());
  float scale = this->scale().convertToFloat();

  popops::scaledAddTo(context.graph, input1, input2, scale, context.seq);
}

void scaledsub_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input1 = context.fromSsa(this->in1());
  poplar::Tensor input2 = context.fromSsa(this->in2());
  float scale = this->scale().convertToFloat();

  popops::scaledSubtractFrom(context.graph, input1, input2, scale, context.seq);
}

void addcmul::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  poplar::Tensor tensor1 = context.fromSsa(this->tensor1());
  poplar::Tensor tensor2 = context.fromSsa(this->tensor2());
  float value = this->value().convertToFloat();

  auto expr = pe::Add(pe::_1, pe::Mul(pe::_3, pe::_2));
  if (value != 1.0f) {
    expr = pe::Add(pe::_1, pe::Mul(pe::Const(value), pe::Mul(pe::_3, pe::_2)));
  }

  poplar::Tensor out =
      popops::map(context.graph, expr, {input, tensor1, tensor2}, context.seq);

  context.addTensor(this->result(), out);
}

void addcmul_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  poplar::Tensor tensor1 = context.fromSsa(this->tensor1());
  poplar::Tensor tensor2 = context.fromSsa(this->tensor2());
  float value = this->value().convertToFloat();

  auto expr = pe::Add(pe::_1, pe::Mul(pe::_3, pe::_2));
  if (value != 1.0f) {
    expr = pe::Add(pe::_1, pe::Mul(pe::Const(value), pe::Mul(pe::_3, pe::_2)));
  }

  popops::mapInPlace(context.graph, expr, {input, tensor1, tensor2},
                     context.seq);
}

void addcdiv::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  poplar::Tensor tensor1 = context.fromSsa(this->tensor1());
  poplar::Tensor tensor2 = context.fromSsa(this->tensor2());
  float value = this->value().convertToFloat();

  auto expr = pe::Add(pe::_1, pe::Divide(pe::_2, pe::_3));
  if (value != 1.0f) {
    expr =
        pe::Add(pe::_1, pe::Mul(pe::Const(value), pe::Divide(pe::_2, pe::_3)));
  }
  poplar::Tensor out =
      popops::map(context.graph, expr, {input, tensor1, tensor2}, context.seq);

  context.addTensor(this->result(), out);
}

void addcdiv_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  poplar::Tensor tensor1 = context.fromSsa(this->tensor1());
  poplar::Tensor tensor2 = context.fromSsa(this->tensor2());
  float value = this->value().convertToFloat();

  auto expr = pe::Add(pe::_1, pe::Divide(pe::_2, pe::_3));
  if (value != 1.0f) {
    expr =
        pe::Add(pe::_1, pe::Mul(pe::Const(value), pe::Divide(pe::_2, pe::_3)));
  }

  popops::mapInPlace(context.graph, expr, {input, tensor1, tensor2},
                     context.seq);
}

void clamp::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor self = context.fromSsa(this->self());

  auto min_optional = this->min();
  auto max_optional = this->max();

  const float min = min_optional.hasValue()
                        ? this->min().getValue().convertToDouble()
                        : std::numeric_limits<float>::lowest();

  const float max = max_optional.hasValue()
                        ? this->max().getValue().convertToDouble()
                        : std::numeric_limits<float>::max();

  auto expr = pe::Clamp(pe::_1, pe::Const(min), pe::Const(max));
  poplar::Tensor out = popops::map(context.graph, expr, {self}, context.seq);

  context.addTensor(this->result(), out);
}

void clampTensor::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor self = context.fromSsa(this->self());
  poplar::Tensor min;
  poplar::Tensor max;
  std::vector<uint64_t> param_shape = {self.shape()[0]};
  // creating numeric limit vectors

  min = this->min() ? context.fromSsa(this->min())
                    : createConstant(context, poplar::FLOAT, param_shape,
                                     std::numeric_limits<float>::lowest());
  max = this->max() ? context.fromSsa(this->max())
                    : createConstant(context, poplar::FLOAT, param_shape,
                                     std::numeric_limits<float>::max());

  auto expr = pe::Clamp(pe::_1, pe::_2, pe::_3);

  poplar::Tensor out =
      popops::map(context.graph, expr, {self, min, max}, context.seq);

  context.addTensor(this->result(), out);
}

void sigmoid_backward::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor grad_output = context.fromSsa(this->grad_output());
  poplar::Tensor output = context.fromSsa(this->output());

  poplar::Tensor out = popnn::nonLinearityInputGradient(
      context.graph, popnn::NonLinearityType::SIGMOID, output, grad_output,
      context.seq);
  context.addTensor(this->grad_input(), out);
}

void tanh_backward::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor grad_output = context.fromSsa(this->grad_output());
  poplar::Tensor output = context.fromSsa(this->output());

  poplar::Tensor out = popnn::nonLinearityInputGradient(
      context.graph, popnn::NonLinearityType::TANH, output, grad_output,
      context.seq);
  context.addTensor(this->grad_input(), out);
}
} // namespace poptorch_ir
