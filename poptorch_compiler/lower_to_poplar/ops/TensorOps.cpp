
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/CompilerHelpers.hpp"
#include <poplar/Graph.hpp>
#include <popops/Fill.hpp>
#include <popops/Zero.hpp>

namespace poptorch_ir {

void print_tensor::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor tensor = context.fromSsa(this->input());
  context.seq.add(poplar::program::PrintTensor(this->title().str(), tensor));
}

void empty_tensor::lowerToPoplar(CompilerContext &context) {
  // Just add the result to the SSA and allocate it if it doesn't exist.
  context.fromSsa(this->result());
}

void fill_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());
  float value = this->value().convertToFloat();

  popops::fill(context.graph, in, context.seq, value);
}

void copy_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor self = context.fromSsa(this->self());
  poplar::Tensor src = context.fromSsa(this->src());
  context.seq.add(poplar::program::Copy(src, self));
}

void zero_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->input());
  popops::zero(context.graph, input, context.seq);
}

void tensorconstant::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->result());

  float as_float = 0.0f;
  // Right now just support 1 value.
  for (mlir::Attribute dimension : this->data()) {
    as_float = dimension.cast<mlir::FloatAttr>().getValueAsDouble();
  }

  popops::fill(context.graph, input, context.seq, as_float);
  //  context.graph.setInitialValue(input, as_float);
}

} // namespace poptorch_ir