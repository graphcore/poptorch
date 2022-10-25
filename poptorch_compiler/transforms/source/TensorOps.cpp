// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLForwardCompat.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <poptorch_logging/Error.hpp>

#include "Helpers.hpp"
#include "dialect/Poptorch.hpp"
#include "dialect/PoptorchDialect.hpp"

namespace poptorch_ir {

#define CANONICALIZE_TO_FULL(input_op, scalar_attr)                            \
  ::mlir::LogicalResult input_op::canonicalize(                                \
      input_op op, ::mlir::PatternRewriter &rewriter) {                        \
    rewriter.replaceOpWithNewOp<full>(op, op->getResultTypes(), scalar_attr);  \
                                                                               \
    return ::mlir::success();                                                  \
  }

CANONICALIZE_TO_FULL(ones, rewriter.getF32FloatAttr(1.0f))
CANONICALIZE_TO_FULL(zeros_like, rewriter.getF32FloatAttr(0.0f))
CANONICALIZE_TO_FULL(full_like,
                     op->getAttrOfType<mlir::FloatAttr>("fill_value"))

::mlir::LogicalResult clone::canonicalize(clone op,
                                          ::mlir::PatternRewriter &rewriter) {
  // If neither the source nor destination are written to after this operation,
  // it is safe to remove the clone operation and just replace dest with src.
  const bool src_has_value_semantics =
      !isOperandView(op.src()) &&
      llvm::none_of(opsThatWriteTo(op.src()), [&op](mlir::Operation *write) {
        return op->isBeforeInBlock(write);
      });
  const bool dest_has_value_semantics = !isWrittenTo(op.result());
  if (dest_has_value_semantics && src_has_value_semantics) {
    rewriter.replaceOp(op, op.src());

    return mlir::success();
  }
  return mlir::failure();
}

::mlir::LogicalResult
overwrite::canonicalize(overwrite op, ::mlir::PatternRewriter &rewriter) {
  // TODO(T64272): currently the eager compiler treats every tensor in the mlir
  // graph as an output. Reinstate this check when this is fixed
  //
  // const auto src_used_after_overwrite = [&] {
  //   return llvm::any_of(op.src().getUses(), isAfter(op));
  // };
  // ERROR_ON(src_used_after_overwrite());

  // Note: we only handle operands who's source is a view here since the other
  // overwrites need to be kept around so the outplace view ops pass does the
  // correct thing
  if (isOperandView(op.src())) {
    op.dest().replaceUsesWithIf(op.src(), isAfter(op));
    rewriter.eraseOp(op);
  }

  return mlir::success();
}

::mlir::LogicalResult cast::canonicalize(cast op,
                                         ::mlir::PatternRewriter &rewriter) {
  const auto self_dtype =
      op.self().getType().cast<::mlir::RankedTensorType>().getElementType();
  const auto result_dtype =
      op.result().getType().cast<::mlir::RankedTensorType>().getElementType();

  // If the dtype of self and result are the same the cast is equivalent to a clone
  if (self_dtype == result_dtype) {
    rewriter.replaceOpWithNewOp<poptorch_ir::clone>(op, op->getResultTypes(), op->getOperands());
    return ::mlir::success();
  }
  return ::mlir::failure();
}

} // namespace poptorch_ir
