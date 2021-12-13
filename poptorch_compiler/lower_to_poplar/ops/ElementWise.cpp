// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/CompilerHelpers.hpp"
#include <popops/ElementWise.hpp>

#include <popops/ScaledAdd.hpp>

namespace pe = popops::expr;

namespace poptorch_ir {

#define BINARY_OP(name)                                                        \
  void name::lowerToPoplar(CompilerContext &context) {                         \
    poplar::Tensor input1 = context.fromSsa(this->in1());                      \
    poplar::Tensor input2 = context.fromSsa(this->in2());                      \
    poplar::Tensor out =                                                       \
        popops::name(context.graph, input1, input2, context.seq);              \
    context.tensors.insert({this->result(), out});                             \
  }                                                                            \
  void name##_::lowerToPoplar(CompilerContext &context) {                      \
    poplar::Tensor input1 = context.fromSsa(this->in1());                      \
    poplar::Tensor input2 = context.fromSsa(this->in2());                      \
    popops::name##InPlace(context.graph, input1, input2, context.seq);         \
  }

#include "binary_ops.h.inc"

#undef BINARY_OP

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

  context.tensors.insert({this->result(), out});
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

  context.tensors.insert({this->result(), out});
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
    context.tensors.insert({this->result(), out});                             \
  }                                                                            \
  void name##_::lowerToPoplar(CompilerContext &context) {                      \
    poplar::Tensor input1 = context.fromSsa(this->in1());                      \
    popops::name##InPlace(context.graph, input1, context.seq);                 \
  }

#include "unary_ops.h.inc"

#undef UNARY_OP

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

  context.tensors.insert({this->result(), out});
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

  context.tensors.insert({this->result(), out});
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

} // namespace poptorch_ir
