// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_MLIR_DISPATCH_UTILS_HPP_
#define POPTORCH_DISPATCH_MLIR_DISPATCH_UTILS_HPP_

#include <c10/core/ScalarType.h>

#include "pytorch_bridge/CompilerTypes.hpp"

namespace poptorch {

c10::ScalarType compilerTypeToScalarType(poptorch_ir::Type type);

} // namespace poptorch

#endif
