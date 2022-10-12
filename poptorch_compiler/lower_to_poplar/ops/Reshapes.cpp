// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <mlir/IR/BuiltinTypes.h>

#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/DynamicSlice.hpp>

#include "../CompilerHelpers.hpp"
#include "dialect/Poptorch.hpp"

namespace poptorch_ir {

void permuteOutplace::lowerToPoplar(poptorch_ir::CompilerContext &context) {
  const poplar::Tensor in = context.fromSsa(this->input());

  auto permutation = convertIntArray<std::uint32_t>(this->dims());

  const poplar::Tensor view = in.dimShuffle(permutation);

  context.addTensor(this->view(), view);
}

void viewOutplace::lowerToPoplar(CompilerContext &context) {
  // Note: despite the name this currently creates an inplace view. This is
  // because the current builder code can never produce a view which cannot be
  // inplaced in this way.
  poplar::Tensor in = context.fromSsa(this->input());

  const std::vector<std::size_t> new_shape =
      convertIntArray<std::size_t>(this->shape());

  // NB intentionally inplace. PyTorch users are told when using reshape
  // that "you should not depend on the copying vs. viewing behavior".
  in = in.reshape(new_shape);
  context.addTensor(result(), in);
}

void expandOutplace::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  // The new rank.
  const std::size_t new_rank = this->shape().size();

  // Add new dims which while we are still smaller than the new tensor.
  while (in.rank() < new_rank) {
    in = in.expand({0});
  }

  for (std::size_t dim = 0; dim < new_rank; ++dim) {
    const std::uint64_t new_dim_int =
        this->shape()[dim].cast<mlir::IntegerAttr>().getUInt();

    // Broadcast dim `new_dim_int` times.
    if (in.shape()[dim] != new_dim_int) {
      in = in.broadcast(new_dim_int, dim);
    }
  }

  context.addTensor(result(), in);
}

void repeat::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  // The new rank.
  const std::size_t new_rank = this->repeats().size();

  while (in.rank() < new_rank) {
    in = in.expand({0});
  }

  for (std::size_t dim = 0; dim < new_rank; dim++) {
    const std::uint64_t new_dim_int =
        this->repeats()[dim].cast<mlir::IntegerAttr>().getUInt();
    in = in.broadcast(new_dim_int, dim);
  }

  context.addTensor(result(), in);
}

namespace {
poplar::Tensor performSlice(const poplar::Tensor &in, int64_t dim,
                            int64_t start, int64_t end, int64_t stride) {
  poplar::Tensor res = in.slice(start, end, dim);
  if (std::abs(stride) != 1) {
    res = res.subSample(std::abs(stride), dim);
  }
  if (stride < 0) {
    res = res.reverse(dim);
  }

  return res;
}
} // namespace

void sliceOutplace::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor in = context.fromSsa(this->input());
  const int64_t dim = this->dim();
  const int64_t start = this->start().getValue();
  const int64_t end = this->end().getValue();
  const int64_t step = this->step();

  context.addTensor(this->result(), performSlice(in, dim, start, end, step));
}

void sliceInverse::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor adapted_view = context.fromSsa(this->view());
  const poplar::Tensor in = context.fromSsa(this->input());
  const int64_t dim = this->dim();
  const int64_t start = this->start().getValue();
  const int64_t end = this->end().getValue();
  const int64_t step = this->step();

  // Note that slice has reference semantics
  auto slice = performSlice(in, dim, start, end, step);
  // For nested slices slice and adapted_view may be the same slice. In this
  // case we don't need to do anything
  if (slice != adapted_view) {
    context.seq.add(poplar::program::Copy(adapted_view, slice));
  }

  context.addTensor(result(), in);
}

void index_select::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor self = context.fromSsa(this->self());
  const poplar::Tensor index = context.fromSsa(this->index())
                                   .flatten()
                                   .expand({1})
                                   .reinterpret(poplar::UNSIGNED_INT);
  const uint64_t dim = this->dim();
  const auto output_size = self.numElements() / self.shape()[dim];

  self = self.dimRoll(dim);
  auto tmp_shape = self.shape();
  self = self.flatten(1, self.rank());

  poplar::OptionFlags options{};
  options.set("usedForSlice", "true");
  options.set("usedForUpdate", "false");
  options.set("operationForUpdate", "none");

  const popops::SlicePlan plan = popops::embedding::plan(
      context.graph, self.elementType(), self.numElements(), output_size,
      {index.numElements()}, options);

  poplar::Tensor out = popops::multiSlice(context.graph, self, index, {0}, {1},
                                          context.seq, plan, options);

  tmp_shape.front() = out.dim(0);

  const auto result_type = result().getType().cast<mlir::RankedTensorType>();
  const auto result_mlir_shape = result_type.getShape();
  const std::vector<std::size_t> result_shape(result_mlir_shape.begin(),
                                              result_mlir_shape.end());

  out = out.reshape(tmp_shape).dimRoll(0, dim).reshape(result_shape);

  context.addTensor(result(), out);
}

} // namespace poptorch_ir
