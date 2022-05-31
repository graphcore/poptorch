// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poplar/Graph.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Fill.hpp>
#include <popops/SortOrder.hpp>
#include <popops/TopK.hpp>
#include <popops/Zero.hpp>
#include <poprand/RandomGen.hpp>

#include "../CompilerHelpers.hpp"

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

void cast::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->self());
  poplar::Tensor out =
      popops::cast(context.graph, in,
                   CompilerContext::poplarTypeOf(this->dtype()), context.seq);
  context.addTensor(this->result(), out);
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

void tensorconstant_float::lowerToPoplar(CompilerContext &context) {
  const std::vector<size_t> shape = convertIntArray<size_t>(this->shape());
  const std::vector<float> data = convertFloatArray<float>(this->data());

  poplar::Tensor new_tensor =
      (data.size() == 1)
          ? createConstant(context, poplar::FLOAT, shape, data[0])
          : createConstantTensor(context, poplar::FLOAT, shape, data);
  context.addTensor(this->result(), new_tensor);
}

void tensorconstant_int::lowerToPoplar(CompilerContext &context) {
  const std::vector<size_t> shape = convertIntArray<size_t>(this->shape());
  const std::vector<int> data = convertIntArray<int>(this->data());

  poplar::Tensor new_tensor =
      (data.size() == 1)
          ? createConstant(context, poplar::INT, shape, data[0])
          : createConstantTensor(context, poplar::INT, shape, data);
  context.addTensor(this->result(), new_tensor);
}

void concat::lowerToPoplar(CompilerContext &context) {
  std::vector<poplar::Tensor> tensors = context.fromSsa(this->tensors());

  context.addTensor(this->result(), poplar::concat(tensors, this->dim()));
}

void topk::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor self = context.fromSsa(this->self());
  const bool largest = this->largest();
  const bool sorted = this->sorted();

  // If sorted is set we should be ranking highest to lowest
  // or lowest to highest depending on whether or not `largest` is set.
  popops::SortOrder order = popops::SortOrder::NONE;
  if (sorted) {
    if (largest) {
      order = popops::SortOrder::DESCENDING;
    } else {
      order = popops::SortOrder::ASCENDING;
    }
  }

  std::uint32_t k = this->K();
  const std::uint32_t dim = this->dim();
  const std::uint32_t last_dim = self.rank() - 1;

  self = self.dimShufflePartial({dim}, {last_dim});

  // Call TopK.
  auto pair = popops::topKWithPermutation(context.graph, context.seq, self,
                                          {k, largest, order});

  poplar::Tensor values = pair.first;
  poplar::Tensor indices = pair.second;

  values = values.dimShufflePartial({dim}, {last_dim});
  indices = indices.dimShufflePartial({dim}, {last_dim});

  std::vector<std::uint64_t> unflattened;

  // Cast to signed int.
  indices = indices.reinterpret(poplar::INT);

  context.addTensor(this->values(), values);
  context.addTensor(this->indices(), indices);
}

void dropout::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor tensor = context.fromSsa(this->input());
  const float p = this->p().convertToFloat();
  const bool training = this->training();

  if (training) {
    // Special case of p=1: return all zeros
    if (p == 1.0) {
      poplar::Tensor result = context.graph.clone(tensor.elementType(), tensor);
      popops::zero(context.graph, result, context.seq);
      context.addTensor(this->result(), result);
      return;
    }
    // NB: Seeds not implemented yet.
    // Need to implement setting seed, seedModifier, and reference tensor
    // TODO(T51096)
    poplar::Tensor result =
        poprand::dropout(context.graph, &context.getRandomSeed(), 0, tensor,
                         tensor, 1. - p, 1. / (1. - p), context.seq);
    context.addTensor(this->result(), result);
  } else {
    context.addTensor(this->result(), tensor);
  }
}

void where::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor condition = context.fromSsa(this->condition());
  poplar::Tensor self = context.fromSsa(this->self());
  poplar::Tensor other = context.fromSsa(this->other());

  auto result =
      popops::select(context.graph, self, other, condition, context.seq);
  context.addTensor(this->result(), result);
}

} // namespace poptorch_ir
