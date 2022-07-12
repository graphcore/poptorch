// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include "dialect/Poptorch.hpp"

namespace poptorch_ir {

// If the return type is a bool convert bitwise ops to logical ops
#define CANONICALIZE_OP_WITH_BOOLEAN_INPUT(inputOp, outputOp)                  \
  ::mlir::LogicalResult inputOp::canonicalize(                                 \
      inputOp op, ::mlir::PatternRewriter &rewriter) {                         \
    const auto result_type = op.getResult().getType();                         \
    const auto result_element_type =                                           \
        result_type.cast<mlir::RankedTensorType>().getElementType();           \
                                                                               \
    if (result_element_type.isInteger(1)) {                                    \
      rewriter.replaceOpWithNewOp<outputOp>(op, op->getOperand(0),             \
                                            op->getOperand(1));                \
                                                                               \
      return mlir::success();                                                  \
    }                                                                          \
    return mlir::failure();                                                    \
  }

CANONICALIZE_OP_WITH_BOOLEAN_INPUT(add, logicalOr)
CANONICALIZE_OP_WITH_BOOLEAN_INPUT(maximum, logicalOr)
CANONICALIZE_OP_WITH_BOOLEAN_INPUT(mul, logicalAnd)
CANONICALIZE_OP_WITH_BOOLEAN_INPUT(minimum, logicalAnd)

CANONICALIZE_OP_WITH_BOOLEAN_INPUT(bitwiseAnd, logicalAnd)
CANONICALIZE_OP_WITH_BOOLEAN_INPUT(bitwiseOr, logicalOr)
CANONICALIZE_OP_WITH_BOOLEAN_INPUT(bitwiseXor, neq)
CANONICALIZE_OP_WITH_BOOLEAN_INPUT(bitwiseXnor, eq)

::mlir::LogicalResult
bitwiseNot::canonicalize(bitwiseNot op, ::mlir::PatternRewriter &rewriter) {
  const auto result_type = op.getResult().getType();
  const auto result_element_type =
      result_type.cast<mlir::RankedTensorType>().getElementType();

  if (result_element_type.isInteger(1)) {
    rewriter.replaceOpWithNewOp<logicalNot>(op, op.getOperand());

    return mlir::success();
  }
  return mlir::failure();
}

::mlir::LogicalResult isnan::canonicalize(isnan op,
                                          ::mlir::PatternRewriter &rewriter) {
  const auto in_type = op.getOperand().getType();
  const auto in_element_type =
      in_type.cast<mlir::RankedTensorType>().getElementType();

  if (in_element_type.isa<mlir::IntegerType>()) {
    rewriter.replaceOpWithNewOp<zeros_like>(op, op->getOperand(0),
                                            rewriter.getIntegerType(1, false));

    return mlir::success();
  }
  return mlir::failure();
}

::mlir::LogicalResult signum::canonicalize(signum op,
                                           ::mlir::PatternRewriter &rewriter) {
  const auto in_type = op.getResult().getType().cast<mlir::RankedTensorType>();
  std::vector<std::int64_t> out_shape(in_type.getShape());
  const auto in_element_type = in_type.getElementType();

  if (in_element_type.isInteger(1)) {
    auto empty =
        rewriter.create<empty_tensor>(op.getLoc(), out_shape, in_element_type);
    rewriter.create<copy_>(op.getLoc(), empty.getResult(), op.getOperand());
    op.getResult().replaceAllUsesWith(empty.getResult());

    return mlir::success();
  }
  return mlir::failure();
}

} // namespace poptorch_ir
