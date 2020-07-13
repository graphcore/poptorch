// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_TO_STRING_HPP
#define INCLUDE_POPTORCH_TO_STRING_HPP

#include <c10/core/ScalarType.h>
#include <string>

namespace poptorch {
inline std::string scalarTypeToOnnxString(const at::ScalarType type) {
  switch (type) {
  case at::ScalarType::Byte:
    return "UINT8";
  case at::ScalarType::Char:
    return "INT8";
  case at::ScalarType::Short:
    return "INT16";
  case at::ScalarType::Int:
    return "INT32";
  case at::ScalarType::Long:
    return "INT64";
  case at::ScalarType::Half:
    return "FLOAT16";
  case at::ScalarType::Float:
    return "FLOAT";
  case at::ScalarType::Double:
    return "DOUBLE";
  case at::ScalarType::ComplexHalf:
    return "UNDEFINED";
  case at::ScalarType::ComplexFloat:
    return "COMPLEX64";
  case at::ScalarType::ComplexDouble:
    return "COMPLEX128";
  case at::ScalarType::Bool:
    return "BOOL";
  case at::ScalarType::QInt8:
    return "UNDEFINED";
  case at::ScalarType::QUInt8:
    return "UNDEFINED";
  case at::ScalarType::QInt32:
    return "UNDEFINED";
  case at::ScalarType::BFloat16:
    return "BFLOAT16";

  default:
    return "(unknown type)";
  }
}

} // namespace poptorch

#endif // INCLUDE_POPTORCH_TO_STRING_HPP
