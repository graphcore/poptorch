// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <torch/csrc/jit/ir/ir.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "PoptorchSymbols.hpp"
#include "popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

namespace {

// An odd function which returns each tensor dimension as an array, a helper for
// torch.max(tensor) and torch.min(tensor). I.E a 4D tensor will return (0, 1,
// 2, 3).
std::vector<std::int64_t>
reduceHelperDimensionCreator(torch::jit::Value *value) {
  // Extract the type from the pytorch IR.
  c10::TensorTypePtr as_tensor = value->type()->expect<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  std::int64_t index = 0;
  // Convert that IR type into a C++ vector of ints.
  std::vector<std::int64_t> shape;
  for (auto optional_int : *dims.sizes()) {
    shape.push_back(index++);
  }
  return shape;
}

class CanonicalizeImpl {
public:
  void run(torch::jit::Graph *graph);

private:
  // This handles the case of both `prim::ListConstruct`
  // and 'prim::Constant[value=[x, y, z]]'.
  template <typename T> std::vector<T> handleList(torch::jit::Node *node);

  template <typename T>
  std::vector<T> handleListConstruct(torch::jit::Node *node);

  // Cast the operand to type T.
  template <typename T>
  torch::jit::Value *handleParamOrConstant(torch::jit::Graph *graph,
                                           torch::jit::Value *operand);

  // Just returns operand if it is a tensor otherwise adds it as a constant and
  // casts it to the right type.
  torch::jit::Value *handleParamOrConstantDeduceType(torch::jit::Graph *graph,
                                                     torch::jit::Value *operand,
                                                     torch::jit::Node *user);

  // Delete a node and also its users if they are also unused.
  void searchAndPossiblyDestroy(torch::jit::Node *node);

  // Fold the constant.
  template <typename T> T foldConstant(torch::jit::Node *node);
};

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

/*
 * ConvertAtenToPopart implementation.
 */

template <typename T> T CanonicalizeImpl::foldConstant(torch::jit::Node *node) {
  // The index of aten::size must be constant.
  std::size_t index = *handleConstant<std::size_t>(node->input(1)->node());

  // Get the shape of the tensor.
  c10::TensorTypePtr as_tensor =
      node->input(0)->type()->expect<c10::TensorType>();
  c10::VaryingShape dims = as_tensor->sizes();

  // Get that requested index.
  return *dims[index];
}

torch::jit::Value *
CanonicalizeImpl::handleParamOrConstantDeduceType(torch::jit::Graph *graph,
                                                  torch::jit::Value *operand,
                                                  torch::jit::Node *user) {
  // The returned value must be a tensor.
  c10::TensorTypePtr return_tensor =
      user->output()->type()->expect<c10::TensorType>();

  // Deduce the type from the scalar type on the return.
  auto optional_scalar_type = return_tensor->scalarType();

  // Means something in the JIT has failed.
  ERROR_ON_MSG(!optional_scalar_type,
               "Internal error: Tensor doesn't have a scalar type.");

  switch (*optional_scalar_type) {
  case c10::ScalarType::Bool:
  case c10::ScalarType::Int:
  case c10::ScalarType::Long: {
    return handleParamOrConstant<std::int32_t>(graph, operand);
  }
  case c10::ScalarType::Float:
  case c10::ScalarType::Double: {
    return handleParamOrConstant<float>(graph, operand);
  }
  default: {
    ERROR("Internal error: Tensor scalar type is unsupported");
  }
  }
}

template <typename T>
torch::jit::Value *
CanonicalizeImpl::handleParamOrConstant(torch::jit::Graph *graph,
                                        torch::jit::Value *operand) {
  torch::jit::Value *value_to_return = operand;
  torch::jit::Node *constant = createIRConstant(graph, operand);

  if (constant) {
    torch::jit::Node *cast = castToType<T>(graph, constant->output());
    value_to_return = cast->output();
  }

  return value_to_return;
}

template <typename T>
std::vector<T> CanonicalizeImpl::handleList(torch::jit::Node *node) {
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
std::vector<T> CanonicalizeImpl::handleListConstruct(torch::jit::Node *node) {
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

void CanonicalizeImpl::searchAndPossiblyDestroy(torch::jit::Node *node) {
  // Skip parameters and nodes with any uses.
  if (node->kind() == c10::prim::Param || node->hasUses()) {
    return;
  }

  // Store the inputs used by this node.
  // Ops may use the same input twice, so use a set to store only unique inputs.
  std::unordered_set<torch::jit::Value *> inputs;
  for (torch::jit::Value *user : node->inputs()) {
    inputs.insert(user);
  }

  // Delete the node.
  node->destroy();

  // If any of the previously used values now have no users repeat the process
  // for them.
  for (torch::jit::Value *user : inputs) {
    searchAndPossiblyDestroy(user->node());
  }
}

void CanonicalizeImpl::run(torch::jit::Graph *graph) {
  for (torch::jit::Node *node : graph->nodes()) {
    logging::LogContext ctx("PopartCanonicalization processing " +
                            nodeToString(node));
    torch::jit::WithInsertPoint insert_point(node);
    torch::jit::Node *new_node = nullptr;
    torch::jit::Symbol kind = node->kind();

    if (SymbolHandler handler = getHandler(node)) {
      new_node = handler(graph, node);
    }

// Handle a constant input.
#define HANDLE(Index, Type) *handleConstant<Type>(node->input(Index)->node())

// Handle an integer dimension attribute (this can be negative hence the special
// case)
#define HANDLE_DIM(Index) handleDimensionParam(node, Index)

#define HANDLE_LIST(Index, Type)                                               \
  handleListConstruct<Type>(node->input(Index)->node())

#define HANDLE_LIST_AS_IR_CONSTANT(Index)                                      \
  intVectorToIrConstant(                                                       \
      graph, handleListConstruct<std::int64_t>(node->input(Index)->node()))

#define HANDLE_TENSOR_LIST(Index) handleTensorList(node->input(Index)->node())
#define PARAM(Index) node->inputs()[Index]
#define COMMA ,
#define NONE

// Only to be used on operations which return a tensor and which this operand
// should be of the same scalar type as the return.
#define PARAM_OR_CONSTANT_ANY_TYPE(Index)                                      \
  handleParamOrConstantDeduceType(graph, node->inputs()[Index], node)

// Add the constant as is but don't try and deduce it to the "right" type as
// that may be ambigous. I.E if an operation takes in an int and a float and
// returns a bool.
#define PARAM_OR_CONSTANT_ANY_TYPE_NO_CAST(Index)                              \
  handleParamOrConstantNoCast(graph, node->inputs()[Index])

// Returns an integer list of dimension that a tensor has. For reduce functions.
// A 5D tensor would return (0, 1, 2, 3, 4)
#define DIMENISON_LENGTH_LIST(Index)                                           \
  reduceHelperDimensionCreator(node->input(Index))

// Returns the shape of the tensor as a vector of ints.
#define TENSOR_SHAPE(Index) shapeFromTensor(node->input(Index))

#define TENSOR_SHAPE_AS_IR(Index) shapeFromTensorAsIR(graph, node->input(Index))

#define GET_RETURN_TYPE getNodeScalarType(node->output())

// Check if the number of inputs is |num|. Used for overload resolution.
#define NUM_INPUTS_EQUALS(Num) node->inputs().size() == Num

#define PARAM_OR_CONSTANT(Index, Type)                                         \
  handleParamOrConstant<Type>(graph, node->inputs()[Index])
// Handle all supported scalar values and pass the correct C++ type to the given
// body.
#define ANY_SCALAR_CONSTANT_HANDLER(Body)                                      \
  at::IntTypePtr type = alphaValue->type()->cast<at::IntType>();               \
  if (type) {                                                                  \
    Body(int) /* NOLINT */                                                     \
  }

// Many binary element wise operations contained a fused "Alpha" component. The
// form of this is A (+, -) B * alpha. Most of the time this will be zero so can
// be skipped but it could be non-zero and must be handled.

// We have a helper macro to insert the alpha value if needed.
#define ALPHA_BODY(Type)                                                       \
  Type alphaAsScalar = *handleConstant<Type>(alphaValue->node());              \
  if (alphaAsScalar != 1) {                                                    \
    torch::jit::Node *alphaConst =                                             \
        CreateConstant<Type>{}(graph, {alphaAsScalar}, {1});                   \
    torch::jit::Node *alphaNode =                                              \
        createMul(graph, {alphaConst->output(), valueToMultiply});             \
    alphaValue = alphaNode->output();                                          \
  }

// If the alpha is 1 we just ignore it otherwise we perform the alpha
// multiplication and use that. This is the macro which should be used.
#define ALPHA(ValueToMultiply, AlphaParam)                                     \
  torch::jit::Value *alphaValue = AlphaParam;                                  \
  torch::jit::Value *valueToMultiply = ValueToMultiply;                        \
  ANY_SCALAR_CONSTANT_HANDLER(ALPHA_BODY)                                      \
  if (alphaValue == (AlphaParam))                                              \
    alphaValue = valueToMultiply;

// Create a function decl with the given call and arguments.
#define OP_CONVERTOR(AtenID, PreBuildCalls, PopartBuilder, Params)             \
  else if (kind == c10::AtenID) { /* NOLINT */                                 \
    PreBuildCalls new_node = PopartBuilder(graph, Params);                     \
  }

// Create a function decl with the given call and arguments.
#define OP_CONVERTOR_WITH_CAST(AtenID, PreBuildCalls, PopartBuilder, Params,   \
                               CastType)                                       \
  else if (kind == c10::AtenID) { /* NOLINT */                                 \
    PreBuildCalls new_node = PopartBuilder(graph, Params);                     \
    new_node = createCast(graph, new_node->output(), CastType);                \
  }

#define OP_CONVERTOR_POP(Sym, PreBuildCalls, PopartBuilder, Params)            \
  else if (kind == Sym) { /* NOLINT */                                         \
    PreBuildCalls newNode = PopartBuilder(graph, Params);                      \
  }
#include "CanonicalizationOps.h.inc"

#undef OP_CONVERTOR
#undef PARAM
#undef COMMA
#undef HANDLE
#undef NONE
#undef ALPHA

    // If we have a new node add it and replace the old use.
    if (new_node) {
      // Mark this node for deletion.
      markNodeForDeletion(node);
      ERROR_ON(node->outputs().size() != new_node->outputs().size());

      if (node->hasUses()) {
        for (std::uint64_t i = 0; i < node->outputs().size(); ++i) {
          replaceOutputUse(node, new_node, i);
        }
      }
    }
  }

  // Build a list of nodes marked for deletion.
  std::unordered_set<torch::jit::Node *> to_delete;
  for (torch::jit::Node *node : graph->nodes()) {
    if (isMarkedForDeletion(node)) {
      to_delete.insert(node);
    }
  }

  // Remove the dead nodes.
  for (torch::jit::Node *node : to_delete) {
    searchAndPossiblyDestroy(node);
  }
}

} // namespace

void canonicalize(torch::jit::Graph *graph) {
  CanonicalizeImpl converter;
  converter.run(graph);
}

} // namespace poptorch
