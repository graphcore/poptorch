// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_UTILS_HPP
#define INCLUDE_POPTORCH_UTILS_HPP

#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <string>
#include <unordered_set>

namespace poptorch {

torch::jit::Node *findEarliestUser(const torch::jit::Value *value);

bool isNondeterministic(const torch::jit::Node &node);

std::string nodeToString(const torch::jit::Node *node);

std::string scalarTypeToOnnxString(at::ScalarType type);

at::ScalarType onnxStrToScalarType(const char *type_str);

// Delete a node and also its users if they are also unused.
void searchAndPossiblyDestroy(
    const std::unordered_set<torch::jit::Node *> &to_test);

// Use unused type BFLOAT16 to indicate ambiguity between FLOAT16 and FLOAT32
// NOLINTNEXTLINE
const auto HALF_OR_FLOAT = at::ScalarType::BFloat16;

std::unique_ptr<char[]> stringToUniquePtr(const std::string &str);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_UTILS_HPP
