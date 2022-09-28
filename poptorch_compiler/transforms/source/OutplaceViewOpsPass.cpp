// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/UseDefLists.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>

#include "Helpers.hpp"
#include "dialect/Poptorch.hpp"
#include "dialect/PoptorchDialect.hpp"

namespace poptorch_ir {

namespace {

auto getOperandIfView(const mlir::Value &operand) {
  return mlir::dyn_cast_or_null<ViewInterface>(operand.getDefiningOp());
}

bool outplaceOperand(mlir::OpOperand &operand,
                     mlir::PatternRewriter &rewriter) {
  if (auto view = getOperandIfView(operand.get())) {
    auto *outplace_view = view.createOutplace(rewriter);
    operand.set(outplace_view->getResult(0));
    return true;
  }
  return false;
}

bool outplaceOperands(mlir::MutableArrayRef<mlir::OpOperand> operands,
                      mlir::PatternRewriter &rewriter) {
  bool changed = false;
  for (auto &&operand : operands) {
    changed |= outplaceOperand(operand, rewriter);
  }
  return changed;
}

struct OutplaceOverwriteOfViewOp final
    : public mlir::OpRewritePattern<overwrite> {
  // Note: this patten has priority over OutplaceViewOps
  explicit OutplaceOverwriteOfViewOp(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<overwrite>(context, 2) {}

  mlir::LogicalResult
  matchAndRewrite(overwrite op,
                  mlir::PatternRewriter &rewriter) const override {
    ERROR_ON(isOperandView(op.src()));
    if (auto view = getOperandIfView(op.dest())) {
      // We have done an inplace operation on a reference view. We need to add
      // a view inverse, to apply the change to the outplace view back to the
      // original tensor
      auto *inverse = view.createInverse(rewriter, op.src());
      llvm::SmallVector<mlir::Value> tensor_operands;
      llvm::copy_if(view->getOperands(), std::back_inserter(tensor_operands),
                    [](const mlir::Value &operand) {
                      return operand.getType().isa<::mlir::RankedTensorType>();
                    });
      auto replacements = llvm::zip(tensor_operands, inverse->getResults());
      for (const auto &[original_operand, inverse_res] : replacements) {
        auto overwrite_op = rewriter.create<overwrite>(
            op->getLoc(), original_operand, inverse_res);
        // Recursing here avoids the need to apply this iteratively
        (void)matchAndRewrite(overwrite_op, rewriter);
      }
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct OutplaceViewOps final : public mlir::RewritePattern {
  explicit OutplaceViewOps(mlir::MLIRContext *context)
      : mlir::RewritePattern(MatchAnyOpTypeTag(), mlir::PatternBenefit(1),
                             context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // We want to keep the original reference view operation around to make it
    // easy to get the correct sequence of outplace view operations and
    // overwrite ops are handled in OutplaceOverwriteOfViewOp
    //
    // Note: we don't bother explicitly deleting the reference view operations.
    // They will be removed by dead code elimination when all the references
    // have been outplaced.
    if (mlir::isa<ViewInterface>(op) || mlir::isa<overwrite>(op)) {
      return mlir::failure();
    }

    // We are using a reference view in an operation. We replace the reference
    // with the outplace view
    //
    // Note: We don't need to do anything clever for return values. If we have a
    // static graph we just want to return the outplace view. In eager mode we
    // will ensure that the output isn't a view elsewhere
    //
    // Note: This will generate unnecessary view operations. These will be
    // cleaned up by the CSE pass (common subexpression elimination)
    bool changed = outplaceOperands(op->getOpOperands(), rewriter);
    return mlir::success(changed);
  }
};

class OutplaceViewOpsPass final
    : public mlir::PassWrapper<OutplaceViewOpsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  OutplaceViewOpsPass() = default;

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.insert<OutplaceOverwriteOfViewOp, OutplaceViewOps>(context);
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOutplaceViewOpsPass() {
  return std::make_unique<OutplaceViewOpsPass>();
}

} // namespace poptorch_ir

static mlir::PassRegistration<poptorch_ir::OutplaceViewOpsPass>
    outplace_view_ops("outplace-view-ops", "");
