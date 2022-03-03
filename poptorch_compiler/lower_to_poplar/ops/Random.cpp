// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "lower_to_poplar/CompilerHelpers.hpp"

#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <poprand/RandomGen.hpp>

#include "poptorch_logging/Error.hpp"

namespace {

// Source: https://pytorch.org/docs/stable/generated/torch.Tensor.random_.html
//
// `torch.random_`'s value limits are defined for ints as the limits of the
// underlying `dtype`, and for floats as 2^mantissa (IEEE754), depending on the
// float size. The minimum defaults to 0.
//
// However, `torch.random_` is implemented in terms of `poprand::uniform`, which
// only accepts a `dtype` of `FLOAT`/`HALF`/`INT`. We also need to convert
// `random_` ranges [from, to) to `poprand::uniform` ranges [from, to].
int32_t getMaxValueForRandom(const poplar::Type &type) {
  if (type == poplar::FLOAT) {
    return 1 << 24;
  }
  if (type == poplar::HALF) {
    return 1 << 11;
  }
  if (type == poplar::INT) {
    return std::numeric_limits<int32_t>::max() - 1;
  }
  if (type == poplar::SHORT) {
    return std::numeric_limits<int16_t>::max() - 1;
  }
  if (type == poplar::UNSIGNED_SHORT) {
    return std::numeric_limits<uint16_t>::max() - 1;
  }
  if (type == poplar::SIGNED_CHAR || type == poplar::CHAR) {
    return std::numeric_limits<int8_t>::max() - 1;
  }
  if (type == poplar::UNSIGNED_CHAR) {
    return std::numeric_limits<uint8_t>::max() - 1;
  }
  if (type == poplar::BOOL) {
    return 1;
  }
  if (type == poplar::UNSIGNED_INT || type == poplar::LONGLONG ||
      type == poplar::UNSIGNED_LONGLONG) {
    ERROR("dtype used for random_ or randint has a too-big largest "
          "representable value (maximum is that of int32).");
  }

  ERROR("Unknown dtype used for random_ or randint.");
}

} // namespace

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

void random_::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor self = context.fromSsa(this->self());

  const int32_t from = 0;
  const int32_t to = getMaxValueForRandom(self.elementType());

  const poplar::Tensor as_ints =
      poprand::uniform(context.graph, &context.getRandomSeed(), 0, self,
                       poplar::INT, from, to, context.seq);

  const poplar::Tensor res =
      popops::cast(context.graph, as_ints, self.elementType(), context.seq);

  context.tensors.insert({this->result(), res});
}

void bernoulli__float::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor input = context.fromSsa(this->input());
  const float prob = this->prob().convertToFloat();

  const poplar::Tensor res =
      poprand::bernoulli(context.graph, &context.getRandomSeed(), 0, input,
                         poplar::FLOAT, prob, context.seq);

  context.tensors.insert({this->result(), res});
}

void bernoulli__tensor::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor input = context.fromSsa(this->input());
  const poplar::Tensor probs = context.fromSsa(this->probs());

  const poplar::Tensor uniform =
      poprand::uniform(context.graph, &context.getRandomSeed(), 0, input,
                       poplar::FLOAT, 0, 1, context.seq);

  const poplar::Tensor as_bools =
      popops::lt(context.graph, uniform, probs, context.seq);

  const poplar::Tensor res =
      popops::cast(context.graph, as_bools, poplar::FLOAT, context.seq);

  context.tensors.insert({this->result(), res});
}

void random__from::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor self = context.fromSsa(this->self());

  const int64_t from = this->from();
  const int64_t to =
      this->to() ? *this->to() -
                       1 // `random_` is [l, h); `poprand::uniform` is [l, h].
                 : getMaxValueForRandom(self.elementType());

  if (self.elementType() == poplar::FLOAT ||
      self.elementType() == poplar::HALF) {
    // Deal with halves & floats directly since they should be allowed to
    // generate values outside of int32 range.
    //
    // Since we're rounding down, the highest number in the range would be
    // extremely unlikely, so add almost 1 -- though not quite enough to make
    // the next number a possibility.
    const poplar::Tensor res =
        poprand::uniform(context.graph, &context.getRandomSeed(), 0, self,
                         self.elementType(), from, to + 0.99, context.seq);
    popops::floorInPlace(context.graph, res, context.seq);

    context.tensors.insert({this->result(), res});
  } else {
    // For everything else, generate the biggest ints we can and cast down
    // if-needed.
    //
    // NOTE: This will defer to the poplar behaviour of failing if passed a
    //       `from`/`to` outside int32 range.
    const poplar::Tensor as_ints =
        poprand::uniform(context.graph, &context.getRandomSeed(), 0, self,
                         poplar::INT, from, to, context.seq);
    const poplar::Tensor res =
        popops::cast(context.graph, as_ints, self.elementType(), context.seq);

    context.tensors.insert({this->result(), res});
  }
}

void bernoulli_out::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor self = context.fromSsa(this->self());

  const poplar::Tensor uniform =
      poprand::uniform(context.graph, &context.getRandomSeed(), 0, self,
                       poplar::FLOAT, 0, 1, context.seq);

  const poplar::Tensor as_bools =
      popops::lt(context.graph, uniform, self, context.seq);

  const poplar::Tensor res =
      popops::cast(context.graph, as_bools, poplar::FLOAT, context.seq);

  context.tensors.insert({this->result(), res});
}

} // namespace poptorch_ir
