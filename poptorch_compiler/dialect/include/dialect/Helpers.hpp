// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_CODEGEN_POPTORCH_DIALECT_HELPERS_H_
#define POPTORCH_CODEGEN_POPTORCH_DIALECT_HELPERS_H_

#include <vector>

#include "mlir/IR/Dialect.h"

namespace poptorch_ir {
std::vector<int64_t> broadcast(const std::vector<int64_t> &lhs,
                               const std::vector<int64_t> &rhs,
                               size_t end_skip = 0);

std::vector<int64_t> getShape(mlir::Value value);

mlir::Type getElementType(mlir::Value value);
} // namespace poptorch_ir

#endif // POPTORCH_CODEGEN_POPTORCH_DIALECT_HELPERS_H_
