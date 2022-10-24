// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "Helpers.hpp"

#include <llvm/ADT/STLExtras.h>

#include "dialect/PoptorchDialect.hpp"

namespace poptorch_ir {

bool isInplace(const mlir::OpOperand &use) {
  auto *owner = use.getOwner();
  if (auto op = mlir::dyn_cast_or_null<overwrite>(owner)) {
    return use.is(op.dest());
  }
  return false;
}

bool isOperandView(const mlir::Value &operand) {
  auto *owner = operand.getDefiningOp();
  return owner != nullptr && owner->hasTrait<mlir::OpTrait::ViewOp>();
}

std::vector<mlir::Operation *> opsThatWriteTo(const mlir::Value &val) {
  std::vector<mlir::Operation *> res;

  for (auto &use : val.getUses()) {
    auto *use_op = use.getOwner();
    if (isInplace(use)) {
      res.push_back(use_op);
    } else if (mlir::isa<ViewInterface>(*use_op)) {
      for (const auto val_view : use_op->getResults()) {
        const auto writes = opsThatWriteTo(val_view);
        res.insert(res.end(), writes.begin(), writes.end());
      }
    }
  }

  return res;
}

bool isWrittenTo(const mlir::Value &val) {
  return !opsThatWriteTo(val).empty();
}

} // namespace poptorch_ir
