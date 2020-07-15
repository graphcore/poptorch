// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poptorch/Utils.hpp"

#include <sstream>

namespace poptorch {

std::string nodeToString(torch::jit::Node *node) {
  std::stringstream ss;
  ss << *node;
  std::string node_str = ss.str();
  return node_str.substr(0, node_str.size() - 1); // Remove trailing line return
}

std::string scalarTypeToOnnxString(const at::ScalarType type) {
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
  case at::ScalarType::BFloat16:
    return "BFLOAT16";
  case at::ScalarType::QInt8:
  case at::ScalarType::QUInt8:
  case at::ScalarType::QInt32:
    return "UNDEFINED";
  default:
    return "(unknown type)";
  }
}
} // namespace poptorch
