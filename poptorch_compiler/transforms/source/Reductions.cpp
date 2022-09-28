// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "dialect/PoptorchDialect.hpp"

namespace poptorch_ir {

template <typename T>
void canonicaliseMinMax(T &op, ::mlir::PatternRewriter &rewriter) {
  auto operand_type = op.getOperand().getType();
  auto size = operand_type.template cast<mlir::RankedTensorType>().getShape();
  if (size.empty()) {
    auto elemtype =
        operand_type.template cast<mlir::RankedTensorType>().getElementType();
    auto empty = rewriter.create<empty_tensor>(op.getLoc(), size, elemtype);
    auto copy =
        rewriter.create<copy_>(op.getLoc(), empty.getResult(), op.getOperand());
    op.getResult(0).replaceAllUsesWith(copy.getResult());
    auto zeros = rewriter.create<zeros_like>(op.getLoc(), op->getResult(1),
                                             std::nullopt);
    op.getResult(1).replaceAllUsesWith(zeros.getResult());
    rewriter.eraseOp(op);
    return;
  }

  auto dim = op.dim();
  auto keepdim = op.keepdim();
  auto topk_result = rewriter.create<topk>(op.getLoc(), op.getOperand(), 1, dim,
                                           std::is_same_v<T, max_dim>, false);
  auto values = topk_result.getResult(0);
  auto indices = topk_result.getResult(1);
  if (!keepdim) {
    values = rewriter.create<squeeze_dim>(op.getLoc(), values, dim).getResult();
    indices =
        rewriter.create<squeeze_dim>(op.getLoc(), indices, dim).getResult();
  }
  op.getResult(0).replaceAllUsesWith(values);
  op.getResult(1).replaceAllUsesWith(indices);
  rewriter.eraseOp(op);
}

::mlir::LogicalResult max_dim::canonicalize(max_dim op,
                                            ::mlir::PatternRewriter &rewriter) {
  canonicaliseMinMax(op, rewriter);
  return mlir::success();
}

::mlir::LogicalResult min_dim::canonicalize(min_dim op,
                                            ::mlir::PatternRewriter &rewriter) {
  canonicaliseMinMax(op, rewriter);
  return mlir::success();
}

::mlir::LogicalResult
median_dim_values::canonicalize(median_dim_values op,
                                ::mlir::PatternRewriter &rewriter) {
  auto operand_type = op.getOperand().getType();
  auto osize = operand_type.cast<mlir::RankedTensorType>().getShape();
  if (osize.empty()) {
    auto elemtype =
        operand_type.template cast<mlir::RankedTensorType>().getElementType();
    auto empty = rewriter.create<empty_tensor>(op.getLoc(), osize, elemtype);
    auto copy =
        rewriter.create<copy_>(op.getLoc(), empty.getResult(), op.getOperand());
    op.getResult(0).replaceAllUsesWith(copy.getResult());
    auto zeros = rewriter.create<zeros_like>(op.getLoc(), op->getResult(1),
                                             std::nullopt);
    op.getResult(1).replaceAllUsesWith(zeros.getResult());
    rewriter.eraseOp(op);
    return mlir::success();
  }

  int64_t dim = op.dim();
  auto keepdim = op.keepdim();
  auto size = osize[dim < 0 ? dim + osize.size() : dim];
  auto half_size = ((size + 1) >> 1);
  auto topk_op = rewriter.create<topk>(op.getLoc(), op.getOperand(), half_size,
                                       dim, false, true);
  auto topk_values = topk_op.getResult(0);
  auto topk_indices = topk_op.getResult(1);

  auto values_slice = rewriter.create<slice_Tensor>(
      op.getLoc(), topk_values, dim, half_size - 1, half_size, 1);

  auto indices_slice = rewriter.create<slice_Tensor>(
      op.getLoc(), topk_indices, dim, half_size - 1, half_size, 1);
  auto values = values_slice.getResult();
  auto indices = indices_slice.getResult();

  if (keepdim) {
    op.getResult(0).replaceAllUsesWith(values);
    op.getResult(1).replaceAllUsesWith(indices);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  auto sq_values =
      rewriter.create<squeeze_dim>(op.getLoc(), values, dim).getResult();
  auto sq_indices =
      rewriter.create<squeeze_dim>(op.getLoc(), indices, dim).getResult();

  op.getResult(0).replaceAllUsesWith(sq_values);
  op.getResult(1).replaceAllUsesWith(sq_indices);

  rewriter.eraseOp(op);
  return mlir::success();
}

} // namespace poptorch_ir
