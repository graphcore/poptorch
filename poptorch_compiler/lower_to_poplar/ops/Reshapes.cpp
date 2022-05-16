// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <vector>

#include "lower_to_poplar/CompilerHelpers.hpp"
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

namespace poptorch_ir {

void reshape::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  std::vector<std::size_t> new_shape =
      convertIntArray<std::size_t>(this->shape());

  // NB intentionally inplace. PyTorch users are told when using reshape
  // that "you should not depend on the copying vs. viewing behavior".
  in = in.reshape(new_shape);
  context.addTensor(result(), in);
}

void squeeze_dim::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  std::int64_t dim = this->dim();

  if (dim < 0) {
    dim += in.rank();
  }
  in = in.squeeze({static_cast<size_t>(dim)});
  context.addTensor(result(), in);
}

void squeeze_dim_::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  std::int64_t dim = this->dim();

  if (dim < 0) {
    dim += in.rank();
  }
  in = in.squeeze({static_cast<size_t>(dim)});
  context.addTensor(result(), in);
}

void unsqueeze::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  std::int64_t dim = this->dim();

  if (dim < 0) {
    // "Negative dim will correspond to unsqueeze() applied at
    // dim = dim + input.dim() + 1."
    // Source:
    // https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
    dim += in.rank() + 1;
  }
  auto shape = in.shape();
  shape.insert(shape.begin() + dim, 1);
  in = in.reshape(shape);
  context.addTensor(result(), in);
}

void expand::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  // The new rank.
  const std::size_t new_rank = this->shape().size();

  // Add new dims which we are still smaller than the new tensor.
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

void as_strided::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  // Lets ignore the stride for now.
  /*std::uint32_t dim = 0;
  // Stride if needed.
  for (mlir::Attribute stride : strides()) {
    const std::uint64_t as_int = stride.cast<mlir::IntegerAttr>().getUInt();

    // Not correct.
    if (as_int > 1) {
      in = in.subSample(as_int, dim++);
    }
  }*/

  std::vector<std::size_t> new_shape =
      convertIntArray<std::size_t>(this->size());
  in = in.reshape(new_shape);

  context.addTensor(result(), in);
}

void transpose::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());
  const std::int64_t dim0 = this->dim0();
  const std::int64_t dim1 = this->dim1();

  // Get 0....N where N is the last dimension index.
  std::vector<std::uint32_t> permutation(in.rank());
  std::iota(permutation.begin(), permutation.end(), 0);

  // Swap them.
  std::swap(permutation[dim0], permutation[dim1]);

  poplar::Tensor view = in.dimShuffle(permutation);

  context.addTensor(result(), view);
}

void select::lowerToPoplar(CompilerContext &context) {
  const poplar::Tensor self = context.fromSsa(this->self());
  const int64_t dim = this->dim();
  const int64_t idx = this->idx();

  poplar::Tensor res = self.slice(idx, idx + 1, dim);

  // The Poplar slice doesn't remove the selected dimension
  // from the shape, whereas Torch select expects it to be
  // removed, this is handled properly in the MLIR shape
  // inference so just resize to the MLIR shape instead of
  // duplicating the logic.
  res = reshapeToMlirShape(res, this->result().getType());

  context.addTensor(this->result(), res);
}

} // namespace poptorch_ir
