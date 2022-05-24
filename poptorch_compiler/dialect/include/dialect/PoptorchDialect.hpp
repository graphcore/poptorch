// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_CODEGEN_POPTORCH_DIALECT_IR_H_
#define POPTORCH_CODEGEN_POPTORCH_DIALECT_IR_H_

#include <vector>

// Dialect.h must be included before PoptorchDialect.h.inc
#include "mlir/IR/Dialect.h"

namespace poptorch_ir {
std::vector<int64_t> broadcast(const std::vector<int64_t> &lhs,
                               const std::vector<int64_t> &rhs,
                               size_t end_skip = 0);
} // namespace poptorch_ir

#include "dialect/PoptorchDialect.h.inc"

#include "Poptorch.hpp"

#endif // POPTORCH_CODEGEN_POPTORCH_DIALECT_IR_H_
