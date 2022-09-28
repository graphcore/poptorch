// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLForwardCompat.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
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

::mlir::LogicalResult copy_::canonicalize(copy_ op,
                                          ::mlir::PatternRewriter &rewriter) {
  if (op.self().getType() != op.src().getType() && op.self().hasOneUse()) {
    // If the destination has a single use we may replace the copy_ with
    // a cast. Otherwise the types being different is a bug. This bug is caught
    // in the verifier for copy_.
    rewriter.replaceOpWithNewOp<cast>(
        op, op.src(),
        op.self().getType().cast<mlir::RankedTensorType>().getElementType());
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

} // namespace poptorch_ir
