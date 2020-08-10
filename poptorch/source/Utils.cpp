// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/ir/ir.h>

#include <sstream>
#include <unordered_set>

#include "PoptorchSymbols.hpp"
#include "poptorch/Utils.hpp"

namespace poptorch {
std::string nodeToString(const torch::jit::Node *node) {
  std::stringstream ss;
  ss << *node;
  std::string node_str = ss.str();
  node_str.pop_back(); // Remove trailing line return
  return node_str;
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

namespace {
bool shouldDestroy(torch::jit::Node *node) {
  // Skip parameters and nodes with any uses.
  return !(node->kind() == c10::prim::Param || node->hasUses());
}

// Store the inputs used by this node.
// Ops may use the same input twice, so use a set to store only unique inputs.
std::unordered_set<torch::jit::Value *> copyInputs(torch::jit::Node *node) {
  std::unordered_set<torch::jit::Value *> inputs;
  for (torch::jit::Value *user : node->inputs()) {
    inputs.insert(user);
  }
  return inputs;
}
} // namespace

void searchAndPossiblyDestroy(torch::jit::Node *node) {
  if (!shouldDestroy(node)) {
    return;
  }

  auto inputs = copyInputs(node);
  node->destroy();

  // If any of the previously used values now have no users repeat the process
  // for them.
  for (auto *user : inputs) {
    searchAndPossiblyDestroy(user->node());
  }
}

void searchAndPossiblyDestroy(torch::jit::graph_node_list_iterator *node_it) {
  torch::jit::Node *node = **node_it;

  if (!shouldDestroy(node)) {
    return;
  }

  auto inputs = copyInputs(node);

  // Delete the node without invalidating iterator
  node_it->destroyCurrent();

  for (auto *user : inputs) {
    searchAndPossiblyDestroy(user->node());
  }
}

} // namespace poptorch
