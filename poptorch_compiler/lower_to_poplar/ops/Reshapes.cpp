// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

#include "../CompilerHelpers.hpp"

namespace poptorch_ir {

void permuteOutplace::lowerToPoplar(poptorch_ir::CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  auto permutation = convertIntArray<std::uint32_t>(this->dims());

  poplar::Tensor view = in.dimShuffle(permutation);

  context.addTensor(this->view(), view);
}

void viewOutplace::lowerToPoplar(CompilerContext &context) {
  // Note: despite the name this currently creates an inplace view. This is
  // because the current builder code can never produce a view which cannot be
  // inplaced in this way.
  poplar::Tensor in = context.fromSsa(this->input());

  std::vector<std::size_t> new_shape =
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

} // namespace poptorch_ir
