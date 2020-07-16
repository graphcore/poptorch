// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "PopartCanonicalizationUtils.hpp"

#include <unordered_map>

#include "poptorch_logging/Logging.hpp"
#include <poptorch/OpBuilder.hpp>
#include <poptorch_logging/Error.hpp>

namespace poptorch {

namespace {

// This avoids the static initialisation order fiasco,
std::unordered_map<c10::Symbol, SymbolHandler> &symbolHandlers() {
  static std::unordered_map<c10::Symbol, SymbolHandler> symbol_handlers;
  return symbol_handlers;
}
/*
 * Helper structs to help deduce the attribute types.
 */

template <typename T> struct Handle {
  template <
      typename = typename std::enable_if<std::is_integral<T>::value>::type>
  std::optional<T> operator()(const c10::Symbol &sym, torch::jit::Node *node) {
    if (node->kindOf(sym) == torch::jit::AttributeKind::i) {
      return node->i(sym);
    }
    if (node->kindOf(sym) == torch::jit::AttributeKind::t) {
      // Sometimes a single long constant is encoded as an at::Tensor.
      const at::Tensor &tensor = node->t(sym);

      if (tensor.sizes().empty()) {
        // Cast tensor to correct value.
        T value = *static_cast<T *>(tensor.data_ptr());
        return value;
      }
    }

    return std::nullopt;
  }
};

template <> struct Handle<float> {
  std::optional<float> operator()(const c10::Symbol &sym,
                                  torch::jit::Node *node) {
    if (node->kindOf(sym) == torch::jit::AttributeKind::f) {
      return node->f(sym);
    }
    if (node->kindOf(sym) == torch::jit::AttributeKind::t) {
      const at::Tensor &value = node->t(sym);
      return *value.data_ptr<double>();
    }
    return std::nullopt;
  }
};

template <> struct Handle<std::vector<std::int64_t>> {
  std::optional<std::vector<std::int64_t>> operator()(const c10::Symbol &sym,
                                                      torch::jit::Node *node) {
    if (node->kindOf(sym) == torch::jit::AttributeKind::is) {
      return node->is(sym);
    }
    return std::nullopt;
  }
};

template <> struct Handle<std::vector<double>> {
  std::optional<std::vector<double>> operator()(const c10::Symbol &sym,
                                                torch::jit::Node *node) {
    if (node->kindOf(sym) == torch::jit::AttributeKind::fs) {
      return node->fs(sym);
    }
    return std::nullopt;
  }
};

// Return true if we know how to fold a given compile time constant operation.
bool canBeConstFolded(torch::jit::Node *node) {
  return node->kind() == c10::aten::size;
}

template <typename T> T foldConstant(torch::jit::Node *node) {
  // The index of aten::size must be constant.
  std::size_t index = *handleConstant<std::size_t>(node->input(1)->node());

  // Get the shape of the tensor.
  c10::TensorTypePtr as_tensor =
      node->input(0)->type()->cast<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  // Get that requested index.
  return *dims[index];
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

template <typename T> std::vector<T> handleList(torch::jit::Node *node) {
  if (node->kind() == c10::prim::ListConstruct) {
    return handleListConstruct<T>(node);
  }
  if (node->kind() == c10::prim::Constant) {
    auto sym = c10::attr::value;

    ERROR_ON_MSG(!node->hasAttribute(sym), "Node must have value attribute");

    return *Handle<std::vector<T>>{}(sym, node);
  }
  std::cerr << "Unhandled list input node:\n";
  node->dump();
  ERROR("List inputs must be of type prim::ListConstruct");
}

template <typename T>
std::vector<T> handleListConstruct(torch::jit::Node *node) {
  ERROR_ON(node->kind() != c10::prim::ListConstruct);

  std::vector<T> result;

  for (torch::jit::Value *value : node->inputs()) {
    std::optional<T> val = handleConstant<T>(value->node());
    if (val) {
      result.push_back(*val);
    }
  }

  return result;
}

template <typename T> std::optional<T> handleConstant(torch::jit::Node *node) {
  // Lists should be explicitly handled in handle list construct.
  if (node->kind() == c10::prim::ListConstruct) {
    return std::nullopt;
  }

  if (node->kind() != c10::prim::Constant && canBeConstFolded(node)) {
    if (std::is_integral<T>::value) {
      return foldConstant<T>(node);
    }
  }

  if (node->kind() != c10::prim::Constant) {
    return std::nullopt;
  }

  auto sym = c10::attr::value;

  if (!node->hasAttribute(sym)) {
    return std::nullopt;
  }

  return Handle<T>{}(sym, node);
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
  std::int64_t dim = *handleConstant<std::int64_t>(node->input(index)->node());

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

torch::jit::Node *createIRConstant(torch::jit::Graph *graph,
                                   torch::jit::Value *value) {
  // Get the scalar type of the result.
  c10::FloatTypePtr as_float = value->type()->cast<c10::FloatType>();
  c10::IntTypePtr as_int = value->type()->cast<c10::IntType>();
  if (as_int) {
    return createConstantInt(
        graph, {*handleConstant<std::int64_t>(value->node())}, {1});
  }
  if (as_float) {
    return createConstantFloat(graph, {*handleConstant<float>(value->node())},
                               {1});
  }

  // If this is still a constant.
  if (value->node()->kind() == c10::prim::Constant) {
    // Scalar doubles and longs are tensors somehow.
    c10::TensorTypePtr as_tensor = value->type()->expect<c10::TensorType>();

    auto sizes = as_tensor->sizes();
    auto type = as_tensor->scalarType();

    if (sizes.size() && *sizes.size() == 0 && type) {
      if (*type == at::kDouble) {
        return createConstantFloat(
            graph, {*handleConstant<float>(value->node())}, {1});
      }
      if (*type == at::kLong) {
        return createConstantInt(
            graph, {*handleConstant<std::int64_t>(value->node())}, {1});
      }
    }

    ERROR("Internal error: Constant type is unsupported");
  }

  // Legal to return null means |value| was not a constant.
  return nullptr;
}

torch::jit::Value *handleParamOrConstantNoCast(torch::jit::Graph *graph,
                                               torch::jit::Value *operand) {
  torch::jit::Value *value_to_return = operand;
  torch::jit::Node *constant = createIRConstant(graph, operand);

  if (constant) {
    value_to_return = constant->output();
  }

  return value_to_return;
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

template std::vector<std::int64_t> handleList(torch::jit::Node *node);
template std::optional<float> handleConstant(torch::jit::Node *);
template std::optional<int> handleConstant(torch::jit::Node *);
template std::optional<bool> handleConstant(torch::jit::Node *);

} // namespace poptorch
