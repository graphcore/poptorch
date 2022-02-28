// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "lower_to_poplar/CompilerHelpers.hpp"

#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <poprand/RandomGen.hpp>

namespace poptorch_ir {

void normal_::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor self = context.fromSsa(this->self());
  const float mean = this->mean().convertToFloat();
  const float stdv = this->stdv().convertToFloat();

  const poplar::Tensor res =
      poprand::normal(context.graph, &context.getRandomSeed(), 0, self,
                      poplar::FLOAT, mean, stdv, context.seq);

  context.tensors.insert({this->result(), res});
}

void normal_Tensor_Tensor::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor means = context.fromSsa(this->means());
  const poplar::Tensor stdvs = context.fromSsa(this->stdvs());

  const poplar::Tensor res =
      poprand::normal(context.graph, &context.getRandomSeed(), 0, means,
                      poplar::FLOAT, 0, 1, context.seq);

  popops::mulInPlace(context.graph, res, stdvs, context.seq);
  popops::addInPlace(context.graph, res, means, context.seq);

  context.tensors.insert({this->result(), res});
}

void normal_Tensor_float::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor means = context.fromSsa(this->means());
  const float stdv = this->stdv().convertToFloat();

  const poplar::Tensor res =
      poprand::normal(context.graph, &context.getRandomSeed(), 0, means,
                      poplar::FLOAT, 0, 1, context.seq);

  popops::mulInPlace(context.graph, res, stdv, context.seq);
  popops::addInPlace(context.graph, res, means, context.seq);

  context.tensors.insert({this->result(), res});
}

void normal_float_Tensor::lowerToPoplar(CompilerContext &context) {
  const float mean = this->mean().convertToFloat();
  const poplar::Tensor stdvs = context.fromSsa(this->stdvs());

  const poplar::Tensor res =
      poprand::normal(context.graph, &context.getRandomSeed(), 0, stdvs,
                      poplar::FLOAT, 0, 1, context.seq);

  popops::mulInPlace(context.graph, res, stdvs, context.seq);
  popops::addInPlace(context.graph, res, mean, context.seq);

  context.tensors.insert({this->result(), res});
}

void uniform_::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor self = context.fromSsa(this->self());
  const float from = this->from().convertToFloat();
  const float to = this->to().convertToFloat();

  const poplar::Tensor res =
      poprand::uniform(context.graph, &context.getRandomSeed(), 0, self,
                       poplar::FLOAT, from, to, context.seq);

  context.tensors.insert({this->result(), res});
}

} // namespace poptorch_ir
