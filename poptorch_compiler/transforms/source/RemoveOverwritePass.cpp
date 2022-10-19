// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "dialect/PoptorchDialect.hpp"

namespace poptorch_ir {

namespace {

auto isAfter(mlir::Operation *op) {
  return [op](const auto &operand) {
    return op->isBeforeInBlock(operand.getOwner());
  };
}

struct RemoveOverwrite final : public mlir::OpRewritePattern<overwrite> {
  explicit RemoveOverwrite(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<overwrite>(context) {}

  mlir::LogicalResult
  matchAndRewrite(overwrite op,
                  mlir::PatternRewriter &rewriter) const override {
    auto new_dest = op.src();

    // If the overwrite doesn't cause a cast, the source of the overwrite isn't
    // used after the overwrite op, and the destination is not an input to the
    // function (inputs to the function have reference semantics), we can just
    // use the src directly without overwriting the destination
    if (op.src().getType() != op.dest().getType()) {
      auto dest_type =
          op.dest().getType().cast<mlir::RankedTensorType>().getElementType();
      auto cast_op = rewriter.create<cast>(op->getLoc(), op.src(), dest_type);
      new_dest = cast_op.result();
    }

    op.dest().replaceUsesWithIf(new_dest, isAfter(op));
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

class RemoveOverwritePass final
    : public mlir::PassWrapper<RemoveOverwritePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  RemoveOverwritePass() = default;

  llvm::StringRef getArgument() const final { return "remove-overwrite"; }

  mlir::StringRef getDescription() const override {
    return "Replace any instances of overwrite ops by just swapping out the "
           "value IDs, in places where that's permissable.";
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.insert<RemoveOverwrite>(context);
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRemoveOverwritePass() {
  return std::make_unique<RemoveOverwritePass>();
}

} // namespace poptorch_ir

static mlir::PassRegistration<poptorch_ir::RemoveOverwritePass>
    remove_redundant_overwrite(poptorch_ir::createRemoveOverwritePass);
