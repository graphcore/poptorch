// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_UTILS_HPP
#define INCLUDE_POPTORCH_UTILS_HPP
#include <torch/csrc/jit/ir/ir.h>

#include <string>

namespace poptorch {

std::string nodeToString(torch::jit::Node *node);

std::string scalarTypeToOnnxString(at::ScalarType type);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_UTILS_HPP
