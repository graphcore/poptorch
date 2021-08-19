// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "pytorch_bridge/PytorchBridgeUtils.hpp"

namespace poptorch_ir {

Type mlirTypeToCompilerType(mlir::Type type) {
  if (type.isUnsignedInteger(8)) {
    return Type::UNSIGNED_CHAR;
  }
  if (type.isSignedInteger(8)) {
    return Type::CHAR;
  }
  if (type.isSignedInteger(16)) {
    return Type::SHORT;
  }
  if (type.isUnsignedInteger(16)) {
    return Type::UNSIGNED_SHORT;
  }
  if (type.isSignedInteger(32)) {
    return Type::INT;
  }
  if (type.isUnsignedInteger(32)) {
    return Type::UNSIGNED_INT;
  }
  if (type.isa<mlir::Float16Type>()) {
    return Type::HALF;
  }
  if (type.isa<mlir::Float32Type>()) {
    return Type::FLOAT;
  }
  if (type.isUnsignedInteger(1)) {
    return Type::BOOL;
  }
  if (type.isa<mlir::BFloat16Type>()) {
    return Type::BFLOAT16;
  }

  return Type::UNDEFINED;
}

} // namespace poptorch_ir