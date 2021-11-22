// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_UTILS_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_UTILS_HPP_
#include <mlir/IR/BuiltinTypes.h>

#include "pytorch_bridge/CompilerTypes.hpp"

namespace poptorch_ir {

Type mlirTypeToCompilerType(mlir::Type type);

} // namespace poptorch_ir

#endif
