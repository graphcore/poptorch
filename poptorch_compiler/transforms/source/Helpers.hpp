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

// What subsequent ops write to the given value, including via views?
//
// TODO(T70702) This is a temporary measure, before explicit
// reference/value-semantics tensor types are implemented.
std::vector<mlir::Operation *> opsThatWriteTo(const mlir::Value &val);

// Is the given value written to later in the program, including if this is done
// via a view?
//
// TODO(T70702) This is a temporary measure, before explicit
// reference/value-semantics tensor types are implemented.
bool isWrittenTo(const mlir::Value &val);

} // namespace poptorch_ir

#endif // POPTORCH_TRANSFORMS_HELPERS_HPP_
