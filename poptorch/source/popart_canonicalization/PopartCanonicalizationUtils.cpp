// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <functional>
#include <numeric>
#include <unordered_map>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/DispatchTracer.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"

#include "../PoptorchSymbols.hpp"

#include "PopartCanonicalizationUtils.hpp"

namespace poptorch {

namespace {

const c10::Symbol delete_node_attr = c10::Symbol::attr("delete_node");

// This avoids the static initialisation order fiasco,
std::unordered_map<c10::Symbol, SymbolHandler> &symbolHandlers() {
  static std::unordered_map<c10::Symbol, SymbolHandler> symbol_handlers;
  return symbol_handlers;
}
} // namespace

bool registerHandler(c10::Symbol symbol, const SymbolHandler &handler) {
  logging::trace("Registering handler for symbol {}", symbol.toDisplayString());
  const bool new_handler = symbolHandlers().emplace(symbol, handler).second;
  ERROR_ON_MSG(!new_handler, "Symbol " << symbol.toDisplayString()
                                       << " already has a handler registered");
  return new_handler;
}

// Return a pointer to a handler if one is registered for this kind of node or
// an empty std::function otherwise.
SymbolHandler getHandler(torch::jit::NodeKind kind) {
  const auto it = symbolHandlers().find(kind);
  if (it != symbolHandlers().cend()) {
    return it->second;
  }
  return {};
}

bool allInputsOfType(torch::jit::Node *node, at::ScalarType type,
                     int ignore_input) {
  int idx = 0;
  for (const auto &input : node->inputs()) {
    if (idx++ == ignore_input) {
      continue;
    }

    const auto tensor_type = input->type()->cast<c10::TensorType>();
    ERROR_ON(!tensor_type);
    ERROR_ON(!tensor_type->scalarType());

    if ((*tensor_type->scalarType()) != type) {
      return false;
    }
  }
  return true;
}

bool allInputsBool(torch::jit::Node *node, int ignore_input) {
  return allInputsOfType(node, at::ScalarType::Bool, ignore_input);
}

bool allInputsInteger(torch::jit::Node *node, int ignore_input) {
  int idx = 0;
  for (const auto &input : node->inputs()) {
    if (idx++ == ignore_input) {
      continue;
    }

    const auto tensor = input->type()->cast<c10::TensorType>();
    ERROR_ON(!tensor);
    ERROR_ON(!tensor->scalarType());

    if (!isIntegralType(*tensor->scalarType(), false)) {
      return false;
    }
  }
  return true;
}

std::vector<torch::jit::Value *> handleTensorList(torch::jit::Node *node) {
  const auto inputs = node->inputs();
  // // Just convert the node->inputs array ref to vector and return it.
  return std::vector<torch::jit::Value *>(inputs.cbegin(), inputs.cend());
}

// Add a vector of ints to the IR as a constant.
torch::jit::Value *
intVectorToIrConstant(torch::jit::Graph *graph,
                      const std::vector<std::int64_t> &ints) {
  const std::vector<std::int64_t> dimensions = {
      static_cast<std::int64_t>(ints.size())};
  return createConstantInt(graph, ints, dimensions)->output();
}

// Get the shape of a tensor and add it to the graph as a constant value.
torch::jit::Value *shapeFromTensorAsIR(torch::jit::Graph *graph,
                                       torch::jit::Value *value) {
  // Extract the type from the pytorch IR.
  const std::vector<std::int64_t> shape = shapeFromTensor(value);
  return intVectorToIrConstant(graph, shape);
}

// Get the scalar type of a given tensor.
at::ScalarType getNodeScalarType(const torch::jit::Value *tensor) {
  // The returned value must be a tensor.
  c10::TensorTypePtr const return_tensor =
      tensor->type()->expect<c10::TensorType>();

  // Deduce the type from the scalar type on the return.
  return *return_tensor->scalarType();
}

bool hasUnityValue(torch::jit::Value *value) {
  const auto tensor = getNodeTensorAttrValue(value->node());
  if (tensor.numel() != 1) {
    return false;
  }
  return tensor.to(at::ScalarType::Float).item<float>() == 1.0;
}

bool isNone(torch::jit::Node *node) {
  if (node->kind() != c10::prim::Constant) {
    return false;
  }

  const auto sym = c10::attr::value;
  return !node->hasAttribute(sym);
}

bool isNone(const torch::jit::Value *value) {
  return value->type()->cast<c10::NoneType>();
}

std::int64_t handleDimensionParam(torch::jit::Value *value,
                                  const c10::TensorTypePtr &as_tensor) {
  // Extract the dim.
  std::int64_t dim = constantToLong(value->node());
  c10::VaryingShape const dims = as_tensor->sizes();

  // If dim is less than zero subtract it to get the actual dimension.
  if (dim < 0) {
    dim = *dims.size() + dim;
  }

  // Return the dim.
  return dim;
}

bool isAnyConstant(torch::jit::Node *node) {
  return isTensorConstant(node) || node->kind() == c10::prim::Constant;
}

bool isFloatingPointConstant(torch::jit::Node *node) {
  const auto tensor_type = node->output()->type()->cast<c10::TensorType>();
  if (tensor_type) {
    const auto scalar_type = *tensor_type->scalarType();
    return c10::isFloatingType(scalar_type);
  }

  ERROR_ON(!node->output()->type()->isSubtypeOf(c10::NumberType::get()));
  return torch::jit::constant_as<at::Scalar>(node->output())
      .value()
      .isFloatingPoint();
}

bool isTensorConstant(torch::jit::Node *node) {
  return (node->kind() == symbols::poptorch::tensor_constant ||
          node->kind() == symbols::poptorch::host_side_tensor_constant);
}

bool isConstantScalar(torch::jit::Value *input) {
  if (!isTensorConstant(input->node())) {
    return false;
  }

  const std::vector<int64_t> shape = shapeFromTensor(input);
  const int64_t numel = std::accumulate(shape.begin(), shape.end(), 1,
                                        std::multiplies<int64_t>());

  return numel == 1;
}

float constantToFloat(torch::jit::Node *node) {
  ERROR_ON_MSG(!isTensorConstant(node), "Cannot force a non-constant '"
                                            << node->kind().toQualString()
                                            << "' node to a float");
  if (node->output()->type()->cast<c10::TensorType>()) {
    return getNodeTensorAttrValue(node).to(at::ScalarType::Float).item<float>();
  }

  ERROR_ON(!node->output()->type()->isSubtypeOf(c10::NumberType::get()));
  const auto s = torch::jit::constant_as<at::Scalar>(node->output());
  return s.value().toFloat();
}

torch::jit::Node *constantToLongConstant(torch::jit::Node *node) {
  ERROR_ON_MSG(!isTensorConstant(node), "Cannot force a non-constant '"
                                            << node->kind().toQualString()
                                            << "' node to a long constant");

  ERROR_ON(!node->output()->type()->cast<c10::TensorType>());
  setNodeTensorAttrValue(node,
                         getNodeTensorAttrValue(node).to(at::ScalarType::Long));
  node->output()->inferTypeFrom(getNodeTensorAttrValue(node));
  return node;
}

std::int32_t constantToInt(torch::jit::Node *node) {
  ERROR_ON_MSG(!isTensorConstant(node), "Cannot force a non-constant '"
                                            << node->kind().toQualString()
                                            << "' node to an int");

  if (node->output()->type()->cast<c10::TensorType>()) {
    return getNodeTensorAttrValue(node)
        .to(at::ScalarType::Int)
        .item<std::int32_t>();
  }

  ERROR_ON(!node->output()->type()->isSubtypeOf(c10::NumberType::get()));
  const auto s = torch::jit::constant_as<at::Scalar>(node->output());
  return s.value().toInt();
}

std::int64_t constantToLong(torch::jit::Node *node) {
  ERROR_ON_MSG(!isTensorConstant(node), "Cannot force a non-constant '"
                                            << node->kind().toQualString()
                                            << "' node to a long");

  if (node->output()->type()->cast<c10::TensorType>()) {
    return getNodeTensorAttrValue(node)
        .to(at::ScalarType::Long)
        .item<std::int64_t>();
  }
  ERROR_ON(!node->output()->type()->isSubtypeOf(c10::NumberType::get()));
  const auto s = torch::jit::constant_as<at::Scalar>(node->output());
  const std::int64_t val = s.value().toLong();

  return val == INT_MAX ? LONG_MAX : val;
}

std::vector<std::int64_t> constantToLongVec(torch::jit::Node *node) {
  return constantListToVec<std::int64_t>(node, constantToLong);
}

std::vector<float> constantToFloatVec(torch::jit::Node *node) {
  return constantListToVec<float>(node, constantToFloat);
}

bool constantToBool(torch::jit::Node *node) {
  ERROR_ON_MSG(!isTensorConstant(node),
               "Cannot force a non-constant node to a bool");

  return constantToInt(node) != 0;
}

std::string constantToString(torch::jit::Node *node) {
  ERROR_ON_MSG(!isTensorConstant(node),
               "Cannot force a non-constant node to a string");

  auto &&t = getNodeTensorAttrValue(node);
  ERROR_ON(!t.is_contiguous());

  const auto length = t.sizes().at(0);
  std::string s(reinterpret_cast<char *>(t.data_ptr()), length);
  return s;
}

at::ScalarType constantToScalarType(torch::jit::Node *node) {
  const auto as_num = constantToInt(node);
  ERROR_ON_MSG(as_num < 0 || as_num > at::NumScalarTypes,
               "Node has a value (" << as_num
                                    << ") which is not "
                                       "representable as a torch dtype");

  return static_cast<at::ScalarType>(as_num);
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
  // Make sure the new value matches the type of the original value.
  new_val->setType(old_val->type());

  // Replace the old value with the new one.
  old_val->replaceAllUsesWith(new_val);
}

void replaceOutputUse(torch::jit::Node *oldNode, torch::jit::Node *new_node,
                      std::uint64_t outputIdx) {
  logging::trace("Replacing node output %{} with that of {}",
                 oldNode->output(outputIdx)->debugName(), *new_node);

  torch::jit::Value *new_val = new_node->output(outputIdx);
  torch::jit::Value *old_val = oldNode->output(outputIdx);
  replaceOutputUse(old_val, new_val);
}

// An odd function which returns each tensor dimension as an array, a helper for
// torch.max(tensor) and torch.min(tensor). I.E a 4D tensor will return (0, 1,
// 2, 3).
std::vector<std::int64_t>
reduceHelperDimensionCreator(torch::jit::Value *value) {
  // Extract the type from the pytorch IR.
  c10::TensorTypePtr const as_tensor = value->type()->expect<c10::TensorType>();
  c10::VaryingShape const dims = as_tensor->sizes();

  // Convert that IR type into a C++ vector of ints.
  std::vector<std::int64_t> shape(dims.sizes()->size());
  // Fill the vector with sequentially incrementing values.
  std::iota(shape.begin(), shape.end(), 0);

  return shape;
}

bool attributeEqual(torch::jit::Node *a, torch::jit::Node *b,
                    c10::Symbol attr) {
  if (!a->hasAttribute(attr) || !b->hasAttribute(attr)) {
    return false;
  }

  const auto attr_kind = a->kindOf(attr);
  if (b->kindOf(attr) != attr_kind) {
    return false;
  }

  switch (attr_kind) {
  case torch::jit::AttributeKind::f:
    return a->f(attr) == b->f(attr);
  case torch::jit::AttributeKind::fs:
    return a->fs(attr) == b->fs(attr);
  case torch::jit::AttributeKind::s:
    return a->s(attr) == b->s(attr);
  case torch::jit::AttributeKind::ss:
    return a->ss(attr) == b->ss(attr);
  case torch::jit::AttributeKind::i:
    return a->i(attr) == b->i(attr);
  case torch::jit::AttributeKind::is:
    return a->is(attr) == b->is(attr);
  case torch::jit::AttributeKind::t:
    return a->t(attr).equal(b->t(attr));
  case torch::jit::AttributeKind::ts: {
    if (a->ts(attr).size() != b->ts(attr).size()) {
      return false;
    }
    auto a_it = a->ts(attr).cbegin();
    auto b_it = b->ts(attr).cbegin();
    for (; a_it != a->ts(attr).cend(); a_it++, b_it++) {
      if (!a_it->equal(*b_it)) {
        return false;
      }
    }
    return true;
  }
  case torch::jit::AttributeKind::g:
    return a->g(attr) == b->g(attr);
  case torch::jit::AttributeKind::gs:
    return a->gs(attr) == b->gs(attr);
  case torch::jit::AttributeKind::c:
    return a->c(attr) == b->c(attr);
  case torch::jit::AttributeKind::cs:
    return a->cs(attr) == b->cs(attr);
  case torch::jit::AttributeKind::ty:
    return a->ty(attr) == b->ty(attr);
  case torch::jit::AttributeKind::tys:
    return a->tys(attr) == b->tys(attr);
  case torch::jit::AttributeKind::ival:
    return a->ival(attr) == b->ival(attr);
  }

  ERROR("Invalid type in attributeSame.");
}

torch::jit::Value *castToPromoteType(torch::jit::Graph *graph,
                                     torch::jit::Value *tensor,
                                     c10::ScalarType promoteType) {
  if (getNodeScalarType(tensor) != promoteType) {
    return createCast(graph, tensor, promoteType)->output();
  }

  return tensor;
}

} // namespace poptorch
