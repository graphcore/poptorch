// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "dialect/PoptorchDialect.hpp"
#include "passes/CommonPasses.hpp"

namespace poptorch_ir {

namespace {

struct BufferizeCopyToGlobalState final
    : public mlir::OpRewritePattern<copy_to_global_state> {
  explicit BufferizeCopyToGlobalState(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<copy_to_global_state>(context) {}

  mlir::LogicalResult
  matchAndRewrite(copy_to_global_state op,
                  mlir::PatternRewriter &rewriter) const override {
    auto tensor_type = op.tensor().getType().cast<mlir::RankedTensorType>();
    auto memref_type = mlir::MemRefType::get(tensor_type.getShape(),
                                             tensor_type.getElementType());

    auto global_memref = rewriter
                             .create<mlir::memref::GetGlobalOp>(
                                 op.getLoc(), memref_type, op.handle())
                             .result();
    rewriter.create<mlir::memref::TensorStoreOp>(op.getLoc(), op.tensor(),
                                                 global_memref);

    rewriter.eraseOp(op);

    return mlir::success();
  }
};

struct BufferizeCopyFromGlobalState final
    : public mlir::OpRewritePattern<copy_from_global_state> {
  explicit BufferizeCopyFromGlobalState(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<copy_from_global_state>(context) {}

  mlir::LogicalResult
  matchAndRewrite(copy_from_global_state op,
                  mlir::PatternRewriter &rewriter) const override {
    auto tensor_type = op.tensor().getType().cast<mlir::RankedTensorType>();
    auto memref_type = mlir::MemRefType::get(tensor_type.getShape(),
                                             tensor_type.getElementType());

    auto global_memref = rewriter
                             .create<mlir::memref::GetGlobalOp>(
                                 op.getLoc(), memref_type, op.handle())
                             .result();
    auto load_val =
        rewriter.create<mlir::memref::LoadOp>(op.getLoc(), global_memref)
            .result();

    auto val =
        rewriter.create<mlir::bufferization::ToTensorOp>(op->getLoc(), load_val)
            .getResult();

    rewriter.replaceOp(op, val);

    return mlir::success();
  }
};

class BufferizeGlobalStatePass final
    : public mlir::PassWrapper<BufferizeGlobalStatePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  BufferizeGlobalStatePass() = default;

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.insert<BufferizeCopyToGlobalState, BufferizeCopyFromGlobalState>(
        context);
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createBufferizeGlobalStatePass() {
  return std::make_unique<BufferizeGlobalStatePass>();
}

} // namespace poptorch_ir
