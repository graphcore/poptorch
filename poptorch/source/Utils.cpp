// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/ir/ir.h>

#include <cstring>
#include <sstream>
#include <unordered_set>

#include "poptorch_logging/Error.hpp"

#include "PoptorchSymbols.hpp"
#include "poptorch/Utils.hpp"

namespace poptorch {

torch::jit::Node *findEarliestUser(const torch::jit::Value *value) {
  auto &uses(value->uses());
  if (uses.empty()) {
    return nullptr;
  }

  torch::jit::Node *earliest_user = uses[0].user;
  for (size_t i = 1; i < uses.size(); i++) {
    auto node = uses[i].user;
    if (node->isBefore(earliest_user)) {
      earliest_user = node;
    }
  }
  return earliest_user;
}

bool isNondeterministic(const torch::jit::Node &node) {
  if (node.isNondeterministic()) {
    return true;
  }

  // Handle extra cases until this is fixed upstream
  // https://github.com/pytorch/pytorch/issues/52599
  if (node.kind() == c10::aten::normal || node.kind() == c10::aten::normal_ ||
      node.kind() == c10::aten::uniform_ ||
      node.kind() == c10::aten::bernoulli_ ||
      node.kind() == c10::aten::feature_dropout) {
    return true;
  }

  return false;
}

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

at::ScalarType onnxStrToScalarType(const char *type_str) {
  if (strcmp(type_str, "UINT8") == 0) {
    return at::ScalarType::Byte;
  }
  if (strcmp(type_str, "INT8") == 0) {
    return at::ScalarType::Char;
  }
  if (strcmp(type_str, "INT16") == 0) {
    return at::ScalarType::Short;
  }
  if (strcmp(type_str, "INT32") == 0) {
    return at::ScalarType::Int;
  }
  if (strcmp(type_str, "INT64") == 0) {
    return at::ScalarType::Long;
  }
  if (strcmp(type_str, "FLOAT16") == 0) {
    return at::ScalarType::Half;
  }
  if (strcmp(type_str, "FLOAT") == 0) {
    return at::ScalarType::Float;
  }
  if (strcmp(type_str, "DOUBLE") == 0) {
    return at::ScalarType::Double;
  }
  if (strcmp(type_str, "COMPLEX64") == 0) {
    return at::ScalarType::ComplexFloat;
  }
  if (strcmp(type_str, "COMPLEX128") == 0) {
    return at::ScalarType::ComplexDouble;
  }
  if (strcmp(type_str, "BOOL") == 0) {
    return at::ScalarType::Bool;
  }
  if (strcmp(type_str, "BFLOAT16") == 0) {
    return at::ScalarType::BFloat16;
  }

  ERROR("No at::scalar_type for " << type_str);
}

namespace {
void processInput(torch::jit::Graph *graph, torch::jit::Value *input,
                  std::vector<torch::jit::Value *> *tensors) {
  switch (input->type()->kind()) {
  case c10::TypeKind::TensorType:
    ERROR_ON(input->node()->kind() != c10::prim::Param &&
             input->node()->kind() != c10::prim::TupleUnpack);
    tensors->push_back(input);
    break;
  case c10::TypeKind::TupleType: {
    // Find the TupleUnpack node
    if (input->hasUses()) {
      ERROR_ON(input->uses().size() != 1);
      auto unpack = input->uses()[0].user;
      ERROR_ON(unpack->kind() != c10::prim::TupleUnpack);
      for (auto element : unpack->outputs()) {
        // Recurse for nested tuple support
        processInput(graph, element, tensors);
      }
    } else {
      // We need placeholders or the values will not align with input tensors
      auto tuple_type = input->type()->expect<c10::TupleType>();
      for (unsigned int i = 0; i < tuple_type->elements().size(); i++) {
        tensors->push_back(nullptr);
      }
    }
    break;
  }

  default:
    ERROR("Unsupported input type '"
          << c10::typeKindToString(input->type()->kind()) << "'");
  }
}
} // namespace

std::vector<torch::jit::Value *>
collapsedGraphInputHierachy(torch::jit::Graph *graph) {
  std::vector<torch::jit::Value *> tensors;

  for (auto *input : graph->inputs()) {
    processInput(graph, input, &tensors);
  }

  return tensors;
}

size_t numTensorsForType(const c10::TypePtr &type) {
  switch (type->kind()) {
  case c10::TypeKind::TensorType:
    return 1;
  case c10::TypeKind::ListType:
    ERROR("Returning a list or tuples of lists is not supported.");
  case c10::TypeKind::TupleType: {
    size_t num_tensors = 0;
    auto tuple = type->expect<c10::TupleType>();
    for (auto &element_type : tuple->elements()) {
      num_tensors += numTensorsForType(element_type);
    }
    return num_tensors;
  }
  default:
    ERROR("Unsupported output type '" << c10::typeKindToString(type->kind())
                                      << "'");
  }
}

namespace {
bool shouldDestroy(torch::jit::Node *node) {
  // Skip parameters and nodes with any uses.
  return !(node->kind() == c10::prim::Param || node->hasUses());
}

// Store the inputs used by this node.
// Ops may use the same input twice, so use a set to store only unique inputs.
std::unordered_set<torch::jit::Node *> copyInputs(torch::jit::Node *node) {
  std::unordered_set<torch::jit::Node *> inputs;
  for (torch::jit::Value *user : node->inputs()) {
    inputs.insert(user->node());
  }
  return inputs;
}

void searchAndPossiblyDestroyInternal(
    torch::jit::Node *node, std::unordered_set<torch::jit::Node *> *destroyed) {
  if (destroyed->count(node)) {
    return;
  }
  if (!shouldDestroy(node)) {
    return;
  }

  auto inputs = copyInputs(node);
  node->destroy();
  destroyed->insert(node);

  // If any of the previously used values now have no users repeat the process
  // for them.
  for (auto *user : inputs) {
    searchAndPossiblyDestroyInternal(user, destroyed);
  }
}
} // namespace

void searchAndPossiblyDestroy(
    const std::unordered_set<torch::jit::Node *> &to_test) {
  std::unordered_set<torch::jit::Node *> destroyed;
  for (auto node : to_test) {
    searchAndPossiblyDestroyInternal(node, &destroyed);
  }
}

std::unique_ptr<char[]> stringToUniquePtr(const std::string &str) {
  auto ptr = std::unique_ptr<char[]>(new char[str.size() + 1]);
  str.copy(ptr.get(), std::string::npos);
  ptr.get()[str.size()] = '\0';
  return ptr;
}

// Convert that IR type into a C++ vector of ints.
std::vector<std::int64_t> shapeFromTensor(torch::jit::Value *value) {
  // Extract the type from the pytorch IR.
  c10::TensorTypePtr as_tensor = value->type()->expect<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  // Convert that IR type into a C++ vector of ints.
  std::vector<std::int64_t> shape;
  if (dims.sizes()) {
    for (auto optional_int : *dims.sizes()) {
      shape.push_back(*optional_int);
    }
  }
  return shape;
}

} // namespace poptorch
