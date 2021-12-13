// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/CompilerHelpers.hpp"
#include <poplar/Graph.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>

namespace poptorch_ir {

void reducemean::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  std::vector<std::size_t> dims = convertIntArray<std::size_t>(this->axes());

  poplar::Tensor output_tensor = popops::reduce(
      context.graph, in, dims, {popops::Operation::ADD}, context.seq);

  // Perfom the inverse divide.
  const int64_t in_elms =
      this->input().getType().cast<mlir::RankedTensorType>().getNumElements();
  const int64_t out_elms =
      this->result().getType().cast<mlir::RankedTensorType>().getNumElements();
  const float ratio =
      static_cast<float>(out_elms) / static_cast<float>(in_elms);
  popops::mulInPlace(context.graph, output_tensor, ratio, context.seq);

  context.tensors.insert({this->result(), output_tensor});
}

void reducesum::lowerToPoplar(CompilerContext &context) {
  poplar::Tensor in = context.fromSsa(this->input());

  std::vector<std::size_t> dims = convertIntArray<std::size_t>(this->axes());

  poplar::Tensor output_tensor = popops::reduce(
      context.graph, in, dims, {popops::Operation::ADD}, context.seq);

  context.tensors.insert({this->result(), output_tensor});
}

} // namespace poptorch_ir
