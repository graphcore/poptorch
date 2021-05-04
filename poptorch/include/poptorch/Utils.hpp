// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_UTILS_HPP
#define INCLUDE_POPTORCH_UTILS_HPP

#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace poptorch {

torch::jit::Node *findEarliestUser(const torch::jit::Value *value);

bool isNondeterministic(const torch::jit::Node &node);

std::string nodeToString(const torch::jit::Node *node);

std::string scalarTypeToOnnxString(at::ScalarType type);

at::ScalarType onnxStrToScalarType(const char *type_str);

// Returns a collapsed version of the graph input hierachy into a list of
// tensor values by following any tuples/lists and their unpacking
// N.B. if a tuple is not used (unpacked), the resulting values will be null
// as a placeholder.
std::vector<torch::jit::Value *>
collapsedGraphInputHierachy(torch::jit::Graph *graph);

// Return the number of tensors for a given type: in the case of a tensor
// this is 1, but in case of nested tuples, this is the sum over all.
size_t numTensorsForType(const c10::TypePtr &type);

// Delete a node and also its users if they are also unused.
void searchAndPossiblyDestroy(
    const std::unordered_set<torch::jit::Node *> &to_test);

// Use unused type BFLOAT16 to indicate ambiguity between FLOAT16 and FLOAT32
// NOLINTNEXTLINE
const auto HALF_OR_FLOAT = at::ScalarType::BFloat16;

std::unique_ptr<char[]> stringToUniquePtr(const std::string &str);

// Get the tensor shape as a vector of ints.
std::vector<std::int64_t> shapeFromTensor(torch::jit::Value *value);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_UTILS_HPP
