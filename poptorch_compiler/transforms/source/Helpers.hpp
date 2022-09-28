// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_TRANSFORMS_HELPERS_HPP_
#define POPTORCH_TRANSFORMS_HELPERS_HPP_

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace poptorch_ir {

inline auto isAfter(mlir::Operation *op) {
  return [op](const auto &operand) {
    return op->isBeforeInBlock(operand.getOwner());
  };
}

bool isInplace(const mlir::OpOperand &use);

bool isOperandView(const mlir::Value &operand);

} // namespace poptorch_ir

#endif // POPTORCH_TRANSFORMS_HELPERS_HPP_
