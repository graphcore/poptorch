// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "Helpers.hpp"

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

} // namespace poptorch_ir
