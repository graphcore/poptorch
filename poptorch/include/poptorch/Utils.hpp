// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_UTILS_HPP
#define INCLUDE_POPTORCH_UTILS_HPP

#include <torch/csrc/jit/ir/ir.h>

#include <string>
#include <unordered_set>

namespace poptorch {

torch::jit::Node *findEarliestUser(const torch::jit::Value *value);

std::string nodeToString(const torch::jit::Node *node);

std::string scalarTypeToOnnxString(at::ScalarType type);

at::ScalarType onnxStrToScalarType(const char *type_str);

// Delete a node and also its users if they are also unused.
void searchAndPossiblyDestroy(
    const std::unordered_set<torch::jit::Node *> &to_test);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_UTILS_HPP
