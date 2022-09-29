// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "pytorch_bridge/PytorchBridgeUtils.hpp"

#include <mlir/IR/BuiltinTypes.h>

#include "poptorch_logging/Error.hpp"

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
  if (type.isa<mlir::NoneType>()) {
    return Type::NONE;
  }

  return Type::UNDEFINED;
}

TensorType mlirTypeToCompilerType(mlir::RankedTensorType type) {
  return TensorType{type.getShape(),
                    mlirTypeToCompilerType(type.getElementType())};
}

TorchReduction getTorchReduction(int reduction) {
  switch (reduction) {
  case 0:
    return TorchReduction::NONE;
  case 1:
    return TorchReduction::MEAN;
  case 2:
    return TorchReduction::SUM;
  default:
    ERROR("Unknown PyTorch reduction " << reduction);
    return TorchReduction::UNKNOWN;
  }
}

} // namespace poptorch_ir
