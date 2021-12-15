
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

#include "lower_to_poplar/CompilerHelpers.hpp"

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

void tensorconstant_float::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->result());

  float as_float = 0.0f;
  // Right now just support 1 value.
  for (mlir::Attribute dimension : this->data()) {
    as_float = dimension.cast<mlir::FloatAttr>().getValueAsDouble();
  }

  popops::fill(context.graph, input, context.seq, as_float);
  //  context.graph.setInitialValue(input, as_float);
}

void tensorconstant_int::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor input = context.fromSsa(this->result());

  int as_int = 0;
  // Right now just support 1 value.
  for (mlir::Attribute dimension : this->data()) {
    as_int = dimension.cast<mlir::IntegerAttr>().getInt();
  }
  popops::fill(context.graph, input, context.seq, as_int);
}

void concat::lowerToPoplar(CompilerContext &context) {
  std::vector<poplar::Tensor> tensors = context.fromSsa(this->tensors());

  context.tensors[this->result()] = poplar::concat(tensors, this->dim());
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

  context.tensors.insert({this->values(), values});
  context.tensors.insert({this->indices(), indices});
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
      context.tensors[this->result()] = result;
      return;
    }
    // NB: Seeds not implemented yet.
    // Need to implement setting seed, seedModifier, and reference tensor
    // TODO(T51096)
    poplar::Tensor seed = context.graph.addVariable(
        poplar::UNSIGNED_INT, {2}, poplar::VariableMappingMethod::LINEAR);
    popops::fill(context.graph, seed, context.seq, 42);
    poplar::Tensor result =
        poprand::dropout(context.graph, &seed, 0, tensor, tensor, 1. - p,
                         1. / (1. - p), context.seq);
    context.tensors[this->result()] = result;
  } else {
    context.tensors[this->result()] = tensor;
  }
}

} // namespace poptorch_ir
