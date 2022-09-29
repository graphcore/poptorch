// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_UTILS_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_UTILS_HPP_
#include <mlir/IR/BuiltinTypes.h>

#include "pytorch_bridge/CompilerTypes.hpp"

namespace poptorch_ir {

Type mlirTypeToCompilerType(mlir::Type type);
TensorType mlirTypeToCompilerType(mlir::RankedTensorType type);

enum class TorchReduction { UNKNOWN = -1, NONE = 0, MEAN = 1, SUM = 2 };

TorchReduction getTorchReduction(int reduction);

} // namespace poptorch_ir

#endif
