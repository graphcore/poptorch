// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <numeric>

#include "dialect/Poptorch.hpp"
#include "dialect/PoptorchDialect.hpp"

namespace poptorch_ir {

::mlir::LogicalResult
transpose::canonicalize(transpose op, ::mlir::PatternRewriter &rewriter) {
  const auto input_type = op.input().getType();
  std::vector<std::int64_t> permutation(
      input_type.cast<mlir::RankedTensorType>().getShape().size());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation.at(op.dim0()), permutation.at(op.dim1()));

  rewriter.replaceOpWithNewOp<permute>(op, op.result().getType(),
                                       op.getOperand(),
                                       rewriter.getI64ArrayAttr(permutation));

  return ::mlir::success();
}

::mlir::LogicalResult
permuteInverse::canonicalize(permuteInverse op,
                             ::mlir::PatternRewriter &rewriter) {
  std::vector<std::int64_t> inverse_permute(op.dims().size());
  for (const auto &ed : llvm::enumerate(op.dims())) {
    const auto value = static_cast<std::size_t>(
        ed.value().cast<mlir::IntegerAttr>().getUInt());
    inverse_permute[value] = ed.index();
  }

  rewriter.replaceOpWithNewOp<permuteOutplace>(
      op, op->getResult(0).getType(), op->getOperand(0),
      rewriter.getI64ArrayAttr(inverse_permute));

  return ::mlir::success();
}

::mlir::LogicalResult
viewInverse::canonicalize(viewInverse op, ::mlir::PatternRewriter &rewriter) {
  const auto result_type = op.result().getType();
  const auto out_shape = result_type.cast<mlir::RankedTensorType>().getShape();
  rewriter.replaceOpWithNewOp<viewOutplace>(
      op, result_type, op->getOperand(0), rewriter.getI64ArrayAttr(out_shape));

  return ::mlir::success();
}

#define CANONICALIZE_OP_INTO_VIEW(inputOp)                                     \
  ::mlir::LogicalResult inputOp::canonicalize(                                 \
      inputOp op, ::mlir::PatternRewriter &rewriter) {                         \
    const auto result_type = op.result().getType();                            \
    const auto out_shape =                                                     \
        result_type.cast<mlir::RankedTensorType>().getShape();                 \
    rewriter.replaceOpWithNewOp<view>(op, op->getOperand(0), out_shape);       \
                                                                               \
    return ::mlir::success();                                                  \
  }

CANONICALIZE_OP_INTO_VIEW(squeeze)
CANONICALIZE_OP_INTO_VIEW(squeeze_dim)
CANONICALIZE_OP_INTO_VIEW(unsqueeze)
// Note: as_strided isn't always a reshape we just haven't implemented it when
// it isn't
CANONICALIZE_OP_INTO_VIEW(as_strided)
CANONICALIZE_OP_INTO_VIEW(alias)
CANONICALIZE_OP_INTO_VIEW(detach)

::mlir::LogicalResult select::canonicalize(select op,
                                           ::mlir::PatternRewriter &rewriter) {
  auto slice = rewriter.create<slice_Tensor>(
      op->getLoc(), op->getOperand(0), op.dim(), op.idx(), op.idx() + 1, 1);
  rewriter.replaceOpWithNewOp<squeeze_dim>(op, slice.result(), op.dim());

  return ::mlir::success();
}

::mlir::LogicalResult
viewOutplace::canonicalize(viewOutplace op, ::mlir::PatternRewriter &rewriter) {
  // If the input and output shape are the same we don't need to do anything
  if (op.input().getType().cast<mlir::RankedTensorType>().getShape() ==
      op.result().getType().cast<mlir::RankedTensorType>().getShape()) {
    rewriter.replaceOp(op, op.input());

    return ::mlir::success();
  }
  return ::mlir::failure();
}

} // namespace poptorch_ir
