// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <type_traits>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <popnn/Loss.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Fill.hpp>
#include <popops/SortOrder.hpp>
#include <popops/TopK.hpp>
#include <popops/Zero.hpp>
#include <poprand/RandomGen.hpp>

#include "../CompilerHelpers.hpp"
#include "dialect/Helpers.hpp"

namespace poptorch_ir {

void print_tensor::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor tensor = context.fromSsa(this->input());
  context.seq.add(poplar::program::PrintTensor(this->title().str(), tensor));
}

void empty_tensor::lowerToPoplar(CompilerContext &context) {
  // Just add the result to the SSA and allocate it if it doesn't exist.
  context.fromSsa(this->result());
}

void cast::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor in = context.fromSsa(this->self());
  const poplar::Tensor out =
      popops::cast(context.graph, in,
                   CompilerContext::poplarTypeOf(this->dtype()), context.seq);
  context.addTensor(this->result(), out);
}

void copy_::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor self = context.fromSsa(this->self());
  const poplar::Tensor src = context.fromSsa(this->src());

  // Note the current way view operations are implemented in lower to poplar can
  // lead to copy_ operations called with the same src and dest
  if (self != src) {
    context.seq.add(poplar::program::Copy(src, self));
  }

  context.addTensor(this->result(), self);
}

void full::lowerToPoplar(CompilerContext &context) {
  const auto out_tensor_type =
      this->result().getType().cast<mlir::RankedTensorType>();
  const auto out_shape = out_tensor_type.getShape();
  const auto out_type =
      CompilerContext::poplarTypeOf(out_tensor_type.getElementType());
  const auto value = fill_value().convertToFloat();

  const auto out = createConstant(context, out_type,
                                  {out_shape.begin(), out_shape.end()}, value);
  context.addTensor(this->result(), out);
}

void tensorconstant_float::lowerToPoplar(CompilerContext &context) {
  const std::vector<size_t> shape = convertIntArray<size_t>(this->shape());
  const std::vector<float> data = convertFloatArray<float>(this->data());

  const poplar::Tensor new_tensor =
      (data.size() == 1)
          ? createConstant(context, poplar::FLOAT, shape, data[0])
          : createConstantTensor(context, poplar::FLOAT, shape, data);
  context.addTensor(this->result(), new_tensor);
}

void tensorconstant_int::lowerToPoplar(CompilerContext &context) {
  const std::vector<size_t> shape = convertIntArray<size_t>(this->shape());
  const std::vector<int> data = convertIntArray<int>(this->data());

  const poplar::Tensor new_tensor =
      (data.size() == 1)
          ? createConstant(context, poplar::INT, shape, data[0])
          : createConstantTensor(context, poplar::INT, shape, data);
  context.addTensor(this->result(), new_tensor);
}

void concat::lowerToPoplar(CompilerContext &context) {
  const std::vector<poplar::Tensor> tensors = context.fromSsa(this->tensors());

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

  const std::uint32_t k = this->K();
  const std::uint32_t dim = this->dim();
  const std::uint32_t last_dim = self.rank() - 1;

  self = self.dimShufflePartial({dim}, {last_dim});

  // Call TopK.
  auto pair = popops::topKWithPermutation(context.graph, context.seq, self,
                                          {k, largest, order});

  poplar::Tensor values = pair.first;
  poplar::Tensor indices = pair.second;

  values = values.dimShufflePartial({last_dim}, {dim});
  indices = indices.dimShufflePartial({last_dim}, {dim});

  // Cast to signed int.
  indices = indices.reinterpret(poplar::INT);

  context.addTensor(this->values(), values);
  context.addTensor(this->indices(), indices);
}

void dropout::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor tensor = context.fromSsa(this->input());
  const float p = this->p().convertToFloat();
  const bool training = this->training();

  if (training) {
    // Special case of p=1: return all zeros
    if (p == 1.0) {
      const poplar::Tensor result =
          context.graph.clone(tensor.elementType(), tensor);
      popops::zero(context.graph, result, context.seq);
      context.addTensor(this->result(), result);
      return;
    }
    // NB: Seeds not implemented yet.
    // Need to implement setting seed, seedModifier, and reference tensor
    // TODO(T51096)
    const poplar::Tensor result =
        poprand::dropout(context.graph, &context.getRandomSeed(), 0, tensor,
                         tensor, 1. - p, 1. / (1. - p), context.seq);
    context.addTensor(this->result(), result);
  } else {
    context.addTensor(this->result(), tensor);
  }
}

void where::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor condition = context.fromSsa(this->condition());
  const poplar::Tensor self = context.fromSsa(this->self());
  const poplar::Tensor other = context.fromSsa(this->other());

  auto result =
      popops::select(context.graph, self, other, condition, context.seq);
  context.addTensor(this->result(), result);
}

namespace {

inline std::vector<std::size_t> reshapeForExtremaOp(poplar::Tensor &input,
                                                    const size_t axis) {
  auto dims = input.shape().size();
  if (dims == 0) {
    // Handle scalar values.
    input = input.reshape({1});
    dims = 1;
  }

  // The axis in which to compute the arg indices should be the last axis in
  // axes. The rest of the axes should be in ascending order.
  std::vector<unsigned int> axes(dims);
  std::iota(axes.begin(), axes.end(), 0);
  axes.erase(axes.begin() + axis);
  axes.push_back(static_cast<unsigned int>(axis));
  input = input.dimShuffle(axes);

  // Reshape the input to a 2D tensor
  auto shape = input.shape();
  const std::size_t dim_0 = std::accumulate(shape.begin(), shape.end() - 1, 1,
                                            std::multiplies<std::size_t>());
  const std::size_t dim_1 = shape.back();
  input = input.reshape({dim_0, dim_1});
  return shape;
}

template <typename Op, typename Func>
inline void extremaOpImpl(Op *op, Func func, CompilerContext &context) {
  poplar::Tensor self = context.fromSsa(op->self());
  const auto keepdim = op->keepdim();
  const auto dim = op->dim().value_or(0);
  if (!op->dim()) {
    self = self.flatten();
  }

  const auto orig_shape = self.shape();
  const auto dimshuf_shape = reshapeForExtremaOp(self, dim);

  auto result = func(context.graph, self, context.seq, "");

  std::vector<std::size_t> new_shape;
  std::copy(dimshuf_shape.begin(), dimshuf_shape.end() - 1,
            std::back_inserter(new_shape));

  if (keepdim) {
    new_shape.insert(new_shape.begin() + dim, 1);
  }

  result = result.reshape(new_shape);
  result = popops::cast(context.graph, result, poplar::INT, context.seq);
  context.addTensor(op->result(), result);
}

} // namespace

void argmin_out::lowerToPoplar(CompilerContext &context) {
  extremaOpImpl(this, popnn::argMin, context);
}

void argmax_out::lowerToPoplar(CompilerContext &context) {
  extremaOpImpl(this, popnn::argMax, context);
}

} // namespace poptorch_ir
