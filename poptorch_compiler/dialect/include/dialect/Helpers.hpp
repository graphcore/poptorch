// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_CODEGEN_POPTORCH_DIALECT_HELPERS_H_
#define POPTORCH_CODEGEN_POPTORCH_DIALECT_HELPERS_H_

#include <string>
#include <vector>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "poptorch_logging/Error.hpp"

namespace poptorch_ir {

inline std::int64_t convertToPositiveDim(std::int64_t dim,
                                         std::size_t numDims) {
  // Pytorch allows you to index into the dimensions of scalar values using 0
  // and -1
  if (numDims == 0) {
    ERROR_ON_MSG(!(dim == -1 || dim == 0),
                 "When indexing into a scalar value the dimension must be in "
                 "the range [-1, 0]");
    return 0;
  }

  if (dim < 0) {
    dim += numDims;
  }

  ERROR_ON_MSG(dim >= static_cast<std::int64_t>(numDims) ||
                   dim < std::int64_t{0},
               "Dimension must be within the range of the tensor");

  return dim;
}

inline std::vector<std::int64_t>
convertToPositiveDim(std::vector<std::int64_t> dim, std::size_t numDims) {
  for (auto &d : dim) {
    d = convertToPositiveDim(d, numDims);
  }

  return dim;
}

inline mlir::RankedTensorType asTensor(const mlir::Value &value) {
  return value.getType().cast<mlir::RankedTensorType>();
}

inline std::vector<int64_t> getShape(const mlir::Value &value) {
  return asTensor(value).getShape();
}

inline mlir::Type getElementType(const mlir::Value &value) {
  return asTensor(value).getElementType();
}

std::vector<int64_t> broadcast(const std::vector<int64_t> &lhs,
                               const std::vector<int64_t> &rhs,
                               size_t end_skip = 0);

inline std::string elementTypeToString(const mlir::Type &type) {
  if (type.isF16()) {
    return "float16";
  }
  if (type.isF32()) {
    return "float32";
  }
  if (type.isUnsignedInteger(8)) {
    return "uint8";
  }
  if (type.isUnsignedInteger(16)) {
    return "uint16";
  }
  if (type.isUnsignedInteger(32)) {
    return "uint32";
  }
  if (type.isUnsignedInteger(64)) {
    return "uint64";
  }
  if (type.isInteger(1)) {
    return "bool";
  }
  if (type.isInteger(8)) {
    return "int8";
  }
  if (type.isInteger(16)) {
    return "int16";
  }
  if (type.isInteger(32)) {
    return "int32";
  }
  if (type.isInteger(64)) {
    return "int64";
  }
  return "Unsupported MLIR type";
}

// Convert any MLIR object to string.
template <typename T> std::string mlirToStr(const T &obj) {
  std::string str;
  {
    llvm::raw_string_ostream ostream(str);
    ostream << obj;
  }
  return str;
}

} // namespace poptorch_ir

#endif // POPTORCH_CODEGEN_POPTORCH_DIALECT_HELPERS_H_
