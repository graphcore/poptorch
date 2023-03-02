// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/ir/ir.h>

#include <cstring>
#include <sstream>
#include <unordered_set>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "PoptorchSymbols.hpp"
#include "popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/Utils.hpp"

namespace poptorch {

torch::jit::Node *findEarliestUser(const torch::jit::Value *value) {
  const auto &uses(value->uses());
  if (uses.empty()) {
    return nullptr;
  }

  torch::jit::Node *earliest_user = uses[0].user;
  for (size_t i = 1; i < uses.size(); i++) {
    auto *node = uses[i].user;
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
  static const auto non_deterministic_nodes = {
      c10::aten::normal,   c10::aten::normal_,   c10::aten::feature_dropout,
      c10::aten::randint,  c10::aten::bernoulli, c10::aten::bernoulli_,
      c10::aten::uniform_, c10::aten::randperm,  c10::aten::exponential_,
      c10::aten::random_,
  };

  return std::find(non_deterministic_nodes.begin(),
                   non_deterministic_nodes.end(),
                   node.kind()) != non_deterministic_nodes.end();
}

std::string nodeToString(const torch::jit::Node *node) {
  std::stringstream ss;
  node->print(ss, 0, nullptr, true, true, false, false);
  std::string node_str = ss.str();
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

at::ScalarType coerceToSupportedType(at::ScalarType type) {
  switch (type) {
  case at::ScalarType::Double:
    return at::ScalarType::Float;
  case at::ScalarType::Long:
    return at::ScalarType::Int;
  default:
    break;
  }
  return type;
}

namespace {
// Adds a null pointers for every unused tensor in an unused tuple
void addNullPtrsForUnusedTuple(const c10::TupleType *tuple_type,
                               std::vector<torch::jit::Value *> *tensors) {
  for (const auto &element : tuple_type->elements()) {
    switch (element->kind()) {
    case c10::TypeKind::TensorType: {
      tensors->push_back(nullptr);
      break;
    }
    case c10::TypeKind::TupleType: {
      auto type = element->expect<c10::TupleType>();
      addNullPtrsForUnusedTuple(type.get(), tensors);
      break;
    }
    default: {
      ERROR("Unsupported input type '" << c10::typeKindToString(element->kind())
                                       << "'");
    }
    }
  }
}

void processInput(torch::jit::Graph *graph, torch::jit::Value *input,
                  std::vector<torch::jit::Value *> *tensors) {
  switch (input->type()->kind()) {
  case c10::TypeKind::TensorType:
    ERROR_ON(input->node()->kind() != c10::prim::Param &&
             input->node()->kind() != c10::prim::TupleUnpack);
    tensors->push_back(input);
    break;

  case c10::TypeKind::ListType: // Fallthrough.
  case c10::TypeKind::TupleType: {
    // Find the TupleUnpack node
    if (input->hasUses()) {
      ERROR_ON(input->uses().size() != 1);
      auto *unpack = input->uses()[0].user;
      ERROR_ON(unpack->kind() != c10::prim::TupleUnpack);
      for (auto *element : unpack->outputs()) {
        // Recurse for nested tuple support
        processInput(graph, element, tensors);
      }
    } else {
      // We need placeholders or the values will not align with input tensors
      auto tuple_type = input->type()->expect<c10::TupleType>();
      addNullPtrsForUnusedTuple(tuple_type.get(), tensors);
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

  case c10::TypeKind::ListType: {
    auto list_type = type->cast<ListTypeWithNumElements>();
    ERROR_ON(!list_type);
    return list_type->numElements();
  }
  case c10::TypeKind::TupleType: {
    size_t num_tensors = 0;
    auto tuple = type->expect<c10::TupleType>();
    for (const auto &element_type : tuple->elements()) {
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
  if (destroyed->count(node) != 0u) {
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
  for (auto *node : to_test) {
    searchAndPossiblyDestroyInternal(node, &destroyed);
  }
}

void removeAndPossiblyDestroyAllInputs(torch::jit::Node *node) {
  std::unordered_set<torch::jit::Node *> inputs;
  for (auto *i : node->inputs()) {
    inputs.insert(i->node());
  }
  node->removeAllInputs();
  searchAndPossiblyDestroy(inputs);
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
  c10::TensorTypePtr const as_tensor = value->type()->expect<c10::TensorType>();
  c10::VaryingShape const dims = as_tensor->sizes();

  // Convert that IR type into a C++ vector of ints.
  std::vector<std::int64_t> shape;
  if (dims.sizes()) {
    for (auto optional_int : *dims.sizes()) {
      shape.push_back(*optional_int);
    }
  }
  return shape;
}

void castWeightAndBias(torch::jit::Graph *graph, torch::jit::Value *input,
                       torch::jit::Value *&weight, torch::jit::Value *&bias) {
  const c10::ScalarType input_type =
      input->type()->expect<c10::TensorType>()->scalarType().value();
  if (!isNone(weight->node())) {
    const c10::ScalarType weight_type =
        weight->type()->expect<c10::TensorType>()->scalarType().value();
    if (weight_type != input_type) {
      weight = createCast(graph, weight, input_type)->output();
    }
  }

  if (!isNone(bias->node())) {
    const c10::ScalarType bias_type =
        bias->type()->expect<c10::TensorType>()->scalarType().value();
    if (bias_type != input_type) {
      bias = createCast(graph, bias, input_type)->output();
    }
  }
}

JitTensorInfo::JitTensorInfo(const at::Tensor &tensor) {
  scalar_type = tensor.scalar_type();
  dims = tensor.sizes().vec();
}

JitTensorInfo::JitTensorInfo(torch::jit::Value *value) {
  auto tensor_type = value->type()->cast<at::TensorType>();
  ERROR_ON_MSG(!tensor_type->scalarType().has_value(), "Data type not set");
  ERROR_ON_MSG(!tensor_type->sizes().concrete_sizes().has_value(),
               "Size not set");
  scalar_type = *tensor_type->scalarType();
  dims = *tensor_type->sizes().concrete_sizes();
}

std::string JitTensorInfo::toString() const {
  std::stringstream ss;
  ss << scalar_type << "(";
  std::string sep;

  for (auto d : dims) {
    ss << sep << d;
    sep = ", ";
  }
  ss << ")";
  return ss.str();
}

void validateTensorShapeAndType(torch::jit::Value *value,
                                const at::Tensor &tensor) {
  JitTensorInfo jit(value);
  JitTensorInfo torch(tensor);
  const bool match = std::tie(torch.scalar_type, torch.dims) ==
                     std::tie(jit.scalar_type, jit.dims);
  ERROR_ON_MSG(!match, "Shape/Type mismatch: JIT tensor %"
                           << value->debugName() << " " << jit.toString()
                           << " is incompatible with " << torch.toString());
}

void setNodeTensorAttrValue(torch::jit::Node *node,
                            torch::jit::TensorAttr::ConstructorType value) {
  node->ts_(c10::attr::value,
            {std::forward<torch::jit::TensorAttr::ConstructorType>(value)});
}

const torch::jit::TensorAttr::ValueType &
getNodeTensorAttrValue(const torch::jit::Node *node) {
  ERROR_ON_MSG(node->kindOf(c10::attr::value) != torch::jit::AttributeKind::ts,
               "[Internal] expected type 'ts' but got "
                   << torch::jit::toString(node->kindOf(c10::attr::value)));
  const auto &ts = node->ts(c10::attr::value);
  ERROR_ON(ts.size() != 1);
  return ts.at(0);
}

} // namespace poptorch
