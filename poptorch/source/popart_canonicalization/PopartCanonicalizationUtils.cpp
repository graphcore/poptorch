// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <functional>
#include <numeric>
#include <unordered_map>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"

#include "../PoptorchSymbols.hpp"

#include "PopartCanonicalizationUtils.hpp"

namespace poptorch {

namespace {

const c10::Symbol delete_node_attr =
    c10::Symbol::fromQualString("attr::delete_node");

// This avoids the static initialisation order fiasco,
std::unordered_map<c10::Symbol, SymbolHandler> &symbolHandlers() {
  static std::unordered_map<c10::Symbol, SymbolHandler> symbol_handlers;
  return symbol_handlers;
}
} // namespace

bool registerHandler(c10::Symbol symbol, const SymbolHandler &handler) {
  logging::trace("Registering handler for symbol {}", symbol.toDisplayString());
  bool new_handler = symbolHandlers().emplace(symbol, handler).second;
  ERROR_ON_MSG(!new_handler, "Symbol " << symbol.toDisplayString()
                                       << " already has a handler registered");
  return new_handler;
}

// Return a pointer to a handler if one is registered for this kind of node or
// an empty std::function otherwise.
SymbolHandler getHandler(torch::jit::Node *node) {
  auto it = symbolHandlers().find(node->kind());
  if (it != symbolHandlers().end()) {
    return it->second;
  }
  return {};
}

std::vector<torch::jit::Value *> handleTensorList(torch::jit::Node *node) {
  std::vector<torch::jit::Value *> result;
  // Just convert the node->inputs array ref to vector and return it.
  for (torch::jit::Value *value : node->inputs()) {
    result.push_back(value);
  }
  return result;
}

// Convert that IR type into a C++ vector of ints.
std::vector<std::int64_t> shapeFromTensor(torch::jit::Value *value) {
  // Extract the type from the pytorch IR.
  c10::TensorTypePtr as_tensor = value->type()->expect<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  // Convert that IR type into a C++ vector of ints.
  std::vector<std::int64_t> shape;
  for (auto optional_int : *dims.sizes()) {
    shape.push_back(*optional_int);
  }
  return shape;
}

// Add a vector of ints to the IR as a constant.
torch::jit::Value *
intVectorToIrConstant(torch::jit::Graph *graph,
                      const std::vector<std::int64_t> &shape) {
  const std::vector<std::int64_t> dimensions = {
      static_cast<std::int64_t>(shape.size())};
  return createConstantInt(graph, shape, dimensions)->output();
}

// Get the shape of a tensor and add it to the graph as a constant value.
torch::jit::Value *shapeFromTensorAsIR(torch::jit::Graph *graph,
                                       torch::jit::Value *value) {
  // Extract the type from the pytorch IR.
  std::vector<std::int64_t> shape = shapeFromTensor(value);
  return intVectorToIrConstant(graph, shape);
}

// Get the scalar type of a given tensor.
at::ScalarType getNodeScalarType(torch::jit::Value *tensor) {
  // The returned value must be a tensor.
  c10::TensorTypePtr return_tensor = tensor->type()->expect<c10::TensorType>();

  // Deduce the type from the scalar type on the return.
  return *return_tensor->scalarType();
}

bool isNone(torch::jit::Node *node) {
  if (node->kind() != c10::prim::Constant) {
    return false;
  }

  auto sym = c10::attr::value;
  return !node->hasAttribute(sym);
}

std::int64_t handleDimensionParam(torch::jit::Node *node, int index) {
  // Extract the dim.
  std::int64_t dim = constantToLong(node->input(index)->node());

  // Get the tensor type. Deduce on the first parameter.
  c10::TensorTypePtr as_tensor =
      node->input(0)->type()->expect<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  // If dim is less than zero subtract it to get the actual dimension.
  if (dim < 0) {
    dim = *dims.size() + dim;
  }

  // Return the dim.
  return dim;
}

float constantToFloat(torch::jit::Node *node) {
  ERROR_ON_MSG(node->kind() != symbols::poptorch::tensor_constant,
               "Cannot force a non-constant node to a float");

  if (node->output()->type()->cast<c10::TensorType>()) {
    return node->t(c10::attr::value).to(at::ScalarType::Float).item<float>();
  }

  ERROR_ON(!node->output()->type()->isSubtypeOf(c10::NumberType::get()));
  auto s = torch::jit::constant_as<at::Scalar>(node->output());
  return s->toFloat();
}

torch::jit::Node *constantToLongConstant(torch::jit::Node *node) {
  ERROR_ON_MSG(node->kind() != symbols::poptorch::tensor_constant,
               "Cannot force a non-constant node to a long constant");
  ERROR_ON(!node->output()->type()->cast<c10::TensorType>());
  node->t_(c10::attr::value,
           node->t(c10::attr::value).to(at::ScalarType::Long));
  node->output()->inferTypeFrom(node->t(c10::attr::value));
  return node;
}

std::int32_t constantToInt(torch::jit::Node *node) {
  ERROR_ON_MSG(node->kind() != symbols::poptorch::tensor_constant,
               "Cannot force a non-constant node to an int");

  if (node->output()->type()->cast<c10::TensorType>()) {
    return node->t(c10::attr::value)
        .to(at::ScalarType::Int)
        .item<std::int32_t>();
  }

  ERROR_ON(!node->output()->type()->isSubtypeOf(c10::NumberType::get()));
  auto s = torch::jit::constant_as<at::Scalar>(node->output());
  return s->toInt();
}

std::int64_t constantToLong(torch::jit::Node *node) {
  ERROR_ON_MSG(node->kind() != symbols::poptorch::tensor_constant,
               "Cannot force a non-constant node to a long");

  if (node->output()->type()->cast<c10::TensorType>()) {
    return node->t(c10::attr::value)
        .to(at::ScalarType::Long)
        .item<std::int64_t>();
  }
  ERROR_ON(!node->output()->type()->isSubtypeOf(c10::NumberType::get()));
  auto s = torch::jit::constant_as<at::Scalar>(node->output());
  std::int64_t val = s->toLong();

  if (val == INT_MAX) {
    return LONG_MAX;
  }

  return val;
}

std::vector<std::int64_t> constantToLongVec(torch::jit::Node *node) {
  ERROR_ON(node->kind() != c10::prim::ListConstruct);

  std::vector<std::int64_t> result;
  for (torch::jit::Value *value : node->inputs()) {
    result.push_back(constantToLong(value->node()));
  }

  return result;
}

bool constantToBool(torch::jit::Node *node) {
  ERROR_ON_MSG(node->kind() != symbols::poptorch::tensor_constant,
               "Cannot force a non-constant node to a bool");

  return constantToInt(node);
}

std::string constantToString(torch::jit::Node *node) {
  ERROR_ON_MSG(node->kind() != symbols::poptorch::tensor_constant,
               "Cannot force a non-constant node to a string");

  auto &&t = node->t(c10::attr::value);
  ERROR_ON(!t.is_contiguous());

  auto length = t.sizes().at(0);
  std::string s(reinterpret_cast<char *>(t.data_ptr()), length);
  return s;
}

std::int32_t convertReduceToPopart(std::int32_t pytorchReduce) {
  // Popart:
  // Sum = 0, Mean =1, NoReduction = 2
  // Pytorch
  // Sum = 2, Mean =1, NoReduction = 0
  if (pytorchReduce == 0) {
    return 2;
  }
  if (pytorchReduce == 1) {
    return 1;
  }
  if (pytorchReduce == 2) {
    return 0;
  }

  ERROR("Unsupported pytorch reduce");
}

void markNodeForDeletion(torch::jit::Node *node) {
  node->i_(delete_node_attr, 1);
}

bool isMarkedForDeletion(torch::jit::Node *node) {
  return node->hasAttribute(delete_node_attr) && node->i(delete_node_attr) > 0;
}

void replaceOutputUse(torch::jit::Value *old_val, torch::jit::Value *new_val) {
  // Take the type of the old value.
  new_val->setType(old_val->type());

  // Replace the old value with the new one.
  old_val->replaceAllUsesWith(new_val);
}

void replaceOutputUse(torch::jit::Node *oldNode, torch::jit::Node *new_node,
                      std::uint64_t outputIdx) {
  torch::jit::Value *new_val = new_node->output(outputIdx);
  torch::jit::Value *old_val = oldNode->output(outputIdx);
  replaceOutputUse(old_val, new_val);
}

} // namespace poptorch
