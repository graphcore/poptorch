// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poptorch/PopartCanonicalization.hpp>

#include <torch/csrc/jit/ir/ir.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <poptorch/OpBuilder.hpp>
#include <poptorch_logging/Error.hpp>
#include <poptorch_logging/Logging.hpp>

#include "PoptorchSymbols.h"

namespace poptorch {

namespace {

// Convert that IR type into a C++ vector of ints.
std::vector<std::int64_t> ShapeFromTensor(torch::jit::Value *value) {
  // Extract the type from the pytorch IR.
  c10::TensorTypePtr asTensor = value->type()->expect<c10::TensorType>();
  c10::VaryingShape dims = asTensor->sizes();

  // Convert that IR type into a C++ vector of ints.
  std::vector<std::int64_t> shape;
  for (auto optionalInt : *dims.sizes()) {
    shape.push_back(*optionalInt);
  }
  return shape;
}

// An odd function which returns each tensor dimension as an array, a helper for
// torch.max(tensor) and torch.min(tensor). I.E a 4D tensor will return (0, 1,
// 2, 3).
std::vector<std::int64_t>
ReduceHelperDimensionCreator(torch::jit::Value *value) {
  // Extract the type from the pytorch IR.
  c10::TensorTypePtr asTensor = value->type()->expect<c10::TensorType>();
  c10::VaryingShape dims = asTensor->sizes();

  std::int64_t index = 0;
  // Convert that IR type into a C++ vector of ints.
  std::vector<std::int64_t> shape;
  for (auto optionalInt : *dims.sizes()) {
    shape.push_back(index++);
  }
  return shape;
}

void ReplaceOutputUse(torch::jit::Value *oldVal, torch::jit::Value *newVal) {
  // Take the type of the old value.
  newVal->setType(oldVal->type());

  // Replace the old value with the new one.
  oldVal->replaceAllUsesWith(newVal);
}

void ReplaceOutputUse(torch::jit::Node *oldNode, torch::jit::Node *newNode,
                      std::uint64_t outputIdx) {
  torch::jit::Value *newVal = newNode->output(outputIdx);
  torch::jit::Value *oldVal = oldNode->output(outputIdx);
  ReplaceOutputUse(oldVal, newVal);
}

class CanonicalizeImpl {
public:
  void Run(torch::jit::Graph &graph);

private:
  // When we transform a node mark it for deletion, this will also clean up
  // unused users afterwards.
  std::unordered_set<torch::jit::Node *> toDelete;

  // This handles the case of both `prim::ListConstruct`
  // and 'prim::Constant[value=[x, y, z]]'.
  template <typename T> std::vector<T> HandleList(torch::jit::Node *node);

  template <typename T>
  std::vector<T> HandleListConstruct(torch::jit::Node *node);

  std::vector<torch::jit::Value *> HandleTensorList(torch::jit::Node *node);

  template <typename T> std::optional<T> HandleConstant(torch::jit::Node *node);

  // Cast the operand to type T.
  template <typename T>
  torch::jit::Value *HandleParamOrConstant(torch::jit::Graph &graph,
                                           torch::jit::Value *operand);

  // Do not cast the operand.
  torch::jit::Value *HandleParamOrConstantNoCast(torch::jit::Graph &graph,
                                                 torch::jit::Value *operand);

  // Just returns operand if it is a tensor otherwise adds it as a constant and
  // casts it to the right type.
  torch::jit::Value *HandleParamOrConstantDeduceType(torch::jit::Graph &graph,
                                                     torch::jit::Value *operand,
                                                     torch::jit::Node *user);

  // Turn a parameter constant into an IR constant.
  torch::jit::Node *CreateIRConstant(torch::jit::Graph &graph,
                                     torch::jit::Value *node);

  std::int64_t HandleDimensionParam(torch::jit::Node *node, int index);

  // Delete a node and also its users if they are also unused.
  void SearchAndPossiblyDestroy(torch::jit::Node *node);

  // Pytorch IR represents 'None' as a constant with no value.
  bool IsNone(torch::jit::Node *node) const;

  // Return true if we know how to fold a given compile time constant operation.
  bool CanBeConstFolded(torch::jit::Node *node);

  // Fold the constant.
  template <typename T> T FoldConstant(torch::jit::Node *node);
};

/*
 * Helper structs to help deduce the attribute types.
 */

template <typename T> struct Handle {
  template <
      typename = typename std::enable_if<std::is_integral<T>::value>::type>
  std::optional<T> operator()(c10::Symbol &sym, torch::jit::Node *node) {
    if (node->kindOf(sym) == torch::jit::AttributeKind::i) {
      return node->i(sym);

    } else if (node->kindOf(sym) == torch::jit::AttributeKind::t) {
      // Sometimes a single long constant is encoded as an at::Tensor.
      at::Tensor tensor = node->t(sym);

      if (tensor.sizes().size() == 0) {
        // Cast tensor to correct value.
        T value = *static_cast<T *>(tensor.data_ptr());
        return value;
      }
    }

    return std::nullopt;
  }
};

template <> struct Handle<float> {
  std::optional<float> operator()(c10::Symbol &sym, torch::jit::Node *node) {
    if (node->kindOf(sym) == torch::jit::AttributeKind::f) {
      return node->f(sym);
    } else if (node->kindOf(sym) == torch::jit::AttributeKind::t) {
      const at::Tensor &value = node->t(sym);
      return *value.data_ptr<double>();
    } else {
      return std::nullopt;
    }
  }
};

template <> struct Handle<std::vector<std::int64_t>> {
  std::optional<std::vector<std::int64_t>> operator()(c10::Symbol &sym,
                                                      torch::jit::Node *node) {
    if (node->kindOf(sym) == torch::jit::AttributeKind::is) {
      return node->is(sym);
    } else {
      return std::nullopt;
    }
  }
};

template <> struct Handle<std::vector<double>> {
  std::optional<std::vector<double>> operator()(c10::Symbol &sym,
                                                torch::jit::Node *node) {
    if (node->kindOf(sym) == torch::jit::AttributeKind::fs) {
      return node->fs(sym);
    } else {
      return std::nullopt;
    }
  }
};

// Both pytorch and popart represent reduce as an enum but with different
// values.
static std::int32_t convertReduceToPopart(std::int32_t pytorchReduce) {
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

/*
 * ConvertAtenToPopart implementation.
 */

// Return true if we know how to fold a given compile time constant operation.
bool CanonicalizeImpl::CanBeConstFolded(torch::jit::Node *node) {
  return node->kind() == c10::aten::size;
}

template <typename T> T CanonicalizeImpl::FoldConstant(torch::jit::Node *node) {
  // The index of aten::size must be constant.
  std::size_t index = *HandleConstant<std::size_t>(node->input(1)->node());

  // Get the shape of the tensor.
  c10::TensorTypePtr asTensor = node->input(0)->type()->cast<c10::TensorType>();
  c10::VaryingShape dims = asTensor->sizes();

  // Get that requested index.
  return *dims[index];
}

// Some operations take in an optional tensor. A "none" constant is passed in to
// mark a tensor which is not there.
bool CanonicalizeImpl::IsNone(torch::jit::Node *node) const {
  if (node->kind() != c10::prim::Constant) {
    return false;
  }

  auto sym = c10::attr::value;
  if (node->hasAttribute(sym)) {
    return false;
  }

  return true;
}

template <typename T>
std::optional<T> CanonicalizeImpl::HandleConstant(torch::jit::Node *node) {
  // Lists should be explicitly handled in handle list construct.
  if (node->kind() == c10::prim::ListConstruct) {
    return std::nullopt;
  }

  if (node->kind() != c10::prim::Constant && CanBeConstFolded(node)) {
    if constexpr (std::is_integral<T>::value) {
      return FoldConstant<T>(node);
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

torch::jit::Value *
CanonicalizeImpl::HandleParamOrConstantDeduceType(torch::jit::Graph &graph,
                                                  torch::jit::Value *operand,
                                                  torch::jit::Node *user) {
  // The returned value must be a tensor.
  c10::TensorTypePtr returnTensor =
      user->output()->type()->expect<c10::TensorType>();

  // Deduce the type from the scalar type on the return.
  auto optionalScalarType = returnTensor->scalarType();

  // Means something in the JIT has failed.
  ERROR_ON_MSG(!optionalScalarType,
               "Internal error: Tensor doesn't have a scalar type.");

  switch (*optionalScalarType) {
  case c10::ScalarType::Bool:
  case c10::ScalarType::Int:
  case c10::ScalarType::Long: {
    return HandleParamOrConstant<std::int32_t>(graph, operand);
  }
  case c10::ScalarType::Float: {
    return HandleParamOrConstant<float>(graph, operand);
  }
  default: {
    ERROR("Internal error: Tensor scalar type is unsupported");
  }
  }
}

torch::jit::Value *
CanonicalizeImpl::HandleParamOrConstantNoCast(torch::jit::Graph &graph,
                                              torch::jit::Value *operand) {
  torch::jit::Value *valueToReturn = operand;
  torch::jit::Node *constant = CreateIRConstant(graph, operand);

  if (constant) {
    valueToReturn = constant->output();
  }

  return valueToReturn;
}

template <typename T>
torch::jit::Value *
CanonicalizeImpl::HandleParamOrConstant(torch::jit::Graph &graph,
                                        torch::jit::Value *operand) {
  torch::jit::Value *valueToReturn = operand;
  torch::jit::Node *constant = CreateIRConstant(graph, operand);

  if (constant) {
    torch::jit::Node *cast = CastToType<T>(graph, constant->output());
    valueToReturn = cast->output();
  }

  return valueToReturn;
}

template <typename T>
std::vector<T> CanonicalizeImpl::HandleList(torch::jit::Node *node) {
  if (node->kind() == c10::prim::ListConstruct) {
    return HandleListConstruct<T>(node);
  } else if (node->kind() == c10::prim::Constant) {
    auto sym = c10::attr::value;

    ERROR_ON_MSG(!node->hasAttribute(sym), "Node must have value attribute");

    return *Handle<std::vector<T>>{}(sym, node);
  }
  std::cerr << "Unhandled list input node:\n";
  node->dump();
  ERROR("List inputs must be of type prim::ListConstruct");
}

std::vector<torch::jit::Value *>
CanonicalizeImpl::HandleTensorList(torch::jit::Node *node) {
  std::vector<torch::jit::Value *> result;

  // Just convert the node->inputs array ref to vector and return it.
  for (torch::jit::Value *value : node->inputs()) {
    result.push_back(value);
  }

  return result;
}

template <typename T>
std::vector<T> CanonicalizeImpl::HandleListConstruct(torch::jit::Node *node) {
  ERROR_ON(node->kind() != c10::prim::ListConstruct);

  std::vector<T> result;

  for (torch::jit::Value *value : node->inputs()) {
    std::optional<T> val = HandleConstant<T>(value->node());
    if (val) {
      result.push_back(*val);
    }
  }

  return result;
}

std::int64_t CanonicalizeImpl::HandleDimensionParam(torch::jit::Node *node,
                                                    int index) {
  // Extract the dim.
  std::int64_t dim = *HandleConstant<std::int64_t>(node->input(index)->node());

  // Get the tensor type. Deduce on the first parameter.
  c10::TensorTypePtr asTensor = node->input(0)->type()->cast<c10::TensorType>();
  c10::VaryingShape dims = asTensor->sizes();

  // If dim is less than zero subtract it to get the actual dimension.
  if (dim < 0) {
    dim = *dims.size() + dim;
  }

  // Return the dim.
  return dim;
}

// Turn a prim::Constant scalar input into a popart graph level scalar constant.
torch::jit::Node *CanonicalizeImpl::CreateIRConstant(torch::jit::Graph &graph,
                                                     torch::jit::Value *value) {
  // Get the scalar type of the result.
  c10::FloatTypePtr asFloat = value->type()->cast<c10::FloatType>();
  c10::IntTypePtr asInt = value->type()->cast<c10::IntType>();
  if (asInt) {
    return Create_ConstantInt(
        graph, {*HandleConstant<std::int64_t>(value->node())}, {1});
  } else if (asFloat) {
    return Create_ConstantFloat(graph, {*HandleConstant<float>(value->node())},
                                {1});
  }

  // If this is still a constant.
  if (value->node()->kind() == c10::prim::Constant) {
    // Scalar doubles and longs are tensors somehow.
    c10::TensorTypePtr asTensor = value->type()->expect<c10::TensorType>();

    auto sizes = asTensor->sizes();
    auto type = asTensor->scalarType();

    if (sizes.size() && *sizes.size() == 0 && type) {
      if (*type == at::kDouble) {
        return Create_ConstantFloat(
            graph, {*HandleConstant<float>(value->node())}, {1});
      } else if (*type == at::kLong) {
        return Create_ConstantInt(
            graph, {*HandleConstant<std::int64_t>(value->node())}, {1});
      }
    }

    ERROR("Internal error: Constant type is unsupported");
  }

  // Legal to return null means |value| was not a constant.
  return nullptr;
}

void CanonicalizeImpl::SearchAndPossiblyDestroy(torch::jit::Node *node) {
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
    SearchAndPossiblyDestroy(user->node());
  }
}

void CanonicalizeImpl::Run(torch::jit::Graph &graph) {
  for (torch::jit::Node *node : graph.nodes()) {
    torch::jit::WithInsertPoint insertPoint(node);
    torch::jit::Node *newNode = nullptr;
    torch::jit::Symbol kind = node->kind();

    // We have a dummy if statement so we can daisy chain the rest of the "else
    // if's" off of it.
    if (kind == c10::aten::view || kind == c10::aten::unsqueeze ||
        kind == c10::aten::flatten || kind == c10::aten::reshape ||
        kind == c10::aten::squeeze) {
      // clang-format off
      // aten::view(Tensor self, int[] size) -> Tensor
      // aten::unsqueeze(Tensor self, int dim) -> Tensor
      // clang-format on

      std::vector<std::int64_t> newShape = ShapeFromTensor(node->output());

      // Reshape the tensor into that shape.
      newNode = CreateReshape(graph, node->inputs()[0], newShape);
    } else if (kind == c10::aten::expand) {
      // clang-format off
      // aten::expand(Tensor self, int[] size)  -> Tensor
      // clang-format on

      // Extract the type from the pytorch IR.
      c10::TensorTypePtr selfTensor =
          node->inputs()[0]->type()->expect<c10::TensorType>();
      c10::VaryingShape selfDims = selfTensor->sizes();

      // Old shape
      std::vector<std::int64_t> oldShape = ShapeFromTensor(node->input(0));

      // Count the elems in the old shape.
      std::int64_t oldElemCount = std::accumulate(
          oldShape.begin(), oldShape.end(), 1, std::multiplies<std::int64_t>());

      // Get the target size for the expand.
      std::vector<std::int64_t> newShape =
          HandleList<int64_t>(node->input(1)->node());

      // Count the number of elements in the target shape.
      std::int64_t newElemCount = std::accumulate(
          newShape.begin(), newShape.end(), 1, std::multiplies<std::int64_t>());

      // Elements don't change so just a reshape.
      if (newElemCount == oldElemCount) {
        newNode = CreateReshape(graph, node->input(0), newShape);
      } else {
        // Otherwise we are expanding the original tensor.
        newNode = Create_ConstantInt(graph, newShape,
                                     {static_cast<int64_t>(newShape.size())});

        newNode = Create_Cast(graph, newNode->output(), c10::kLong);
        newNode = Create_expand(graph, {node->input(0), newNode->output()});
      }
    } else if (kind == c10::aten::expand_as) {
      // clang-format off
      // aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor
      // aten::expand_as(Tensor self, Tensor other) -> Tensor
      // clang-format on

      // Extract the type from the pytorch IR.
      c10::TensorTypePtr selfTensor =
          node->input(0)->type()->expect<c10::TensorType>();
      c10::VaryingShape selfDims = selfTensor->sizes();

      std::int64_t oldElemCount = 0;
      for (auto optionalInt : *selfDims.sizes()) {
        oldElemCount += *optionalInt;
      }

      // Extract the type from the pytorch IR.
      c10::TensorTypePtr asTensor =
          node->input(1)->type()->expect<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();

      // Convert that IR type into a C++ vector of ints.
      std::vector<std::int64_t> newShape;
      std::int64_t newElemCount = 0;

      for (auto optionalInt : *dims.sizes()) {
        newShape.push_back(*optionalInt);
        newElemCount += *optionalInt;
      }

      // Elements don't change so just a reshape.
      if (newElemCount == oldElemCount) {
        newNode = CreateReshape(graph, node->input(0), newShape);
      } else {
        newNode = Create_ConstantInt(graph, newShape,
                                     {static_cast<int64_t>(newShape.size())});

        newNode = Create_Cast(graph, newNode->output(), c10::kLong);

        newNode = Create_expand(graph, {node->input(0), newNode->output()});
      }
    }

// Handle a constant input.
#define HANDLE(Index, Type) *HandleConstant<Type>(node->input(Index)->node())

// Handle an integer dimension attribute (this can be negative hence the special
// case)
#define HANDLE_DIM(Index) HandleDimensionParam(node, Index)

#define HANDLE_LIST(Index, Type)                                               \
  HandleListConstruct<Type>(node->input(Index)->node())
#define HANDLE_TENSOR_LIST(Index) HandleTensorList(node->input(Index)->node())
#define PARAM(Index) node->inputs()[Index]
#define COMMA ,
#define NONE

// Only to be used on operations which return a tensor and which this operand
// should be of the same scalar type as the return.
#define PARAM_OR_CONSTANT_ANY_TYPE(Index)                                      \
  HandleParamOrConstantDeduceType(graph, node->inputs()[Index], node)

// Add the constant as is but don't try and deduce it to the "right" type as
// that may be ambigous. I.E if an operation takes in an int and a float and
// returns a bool.
#define PARAM_OR_CONSTANT_ANY_TYPE_NO_CAST(Index)                              \
  HandleParamOrConstantNoCast(graph, node->inputs()[Index])

// Returns an integer list of dimension that a tensor has. For reduce functions.
// A 5D tensor would return (0, 1, 2, 3, 4)
#define DIMENISON_LENGTH_LIST(Index)                                           \
  ReduceHelperDimensionCreator(node->inputs()[Index])

// Check if the number of inputs is |num|. Used for overload resolution.
#define NUM_INPUTS_EQUALS(Num) node->inputs().size() == Num

#define PARAM_OR_CONSTANT(Index, Type)                                         \
  HandleParamOrConstant<Type>(graph, node->inputs()[Index])
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
  Type alphaAsScalar = *HandleConstant<Type>(alphaValue->node());              \
  if (alphaAsScalar != 1) {                                                    \
    torch::jit::Node *alphaConst =                                             \
        Create_Constant<Type>{}(graph, {alphaAsScalar}, {1});                  \
    torch::jit::Node *alphaNode =                                              \
        Create_mul(graph, {alphaConst->output(), valueToMultiply});            \
    alphaValue = alphaNode->output();                                          \
  }

// If the alpha is 1 we just ignore it otherwise we perform the alpha
// multiplication and use that. This is the macro which should be used
#define ALPHA(ValueToMultiply, AlphaParam)                                     \
  torch::jit::Value *alphaValue = AlphaParam;                                  \
  torch::jit::Value *valueToMultiply = ValueToMultiply;                        \
  ANY_SCALAR_CONSTANT_HANDLER(ALPHA_BODY)                                      \
  if (alphaValue == AlphaParam)                                                \
    alphaValue = valueToMultiply;

// Create a function decl with the given call and arguments.
#define OP_CONVERTOR(AtenID, PreBuildCalls, PopartBuilder, Params)             \
  else if (kind == c10::AtenID) { /* NOLINT */                                 \
    PreBuildCalls newNode = PopartBuilder(graph, Params);                      \
  }

#include "CanonicalizationOps.h.inc"

#undef OP_CONVERTOR
#undef PARAM
#undef COMMA
#undef HANDLE
#undef NONE
#undef ALPHA

    // NOLINTNEXTLINE
    else if (kind == c10::aten::_convolution) {
      // clang-format off
      /*
      aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[]
      stride, int[] padding, int[] dilation, bool transposed, int[]
      output_padding, int groups) -> Tensor
      */
      // clang-format on
      std::optional<std::int64_t> transposed =
          HandleConstant<std::int64_t>(node->input(6)->node());

      torch::jit::Value *input = node->input(0);
      torch::jit::Value *kernel = node->input(1);

      std::vector<torch::jit::Value *> inputs{input, kernel};

      if (!IsNone(node->input(2)->node())) {
        inputs.push_back(node->input(2));
      }

      std::vector<std::int64_t> stride =
          HandleList<int64_t>(node->input(3)->node());
      std::vector<std::int64_t> padding =
          HandleList<std::int64_t>(node->input(4)->node());

      // Pytorch gives the padding as being the amount to pad in both
      // directions. Popart two arguments for each axis, the amount to pad in
      // each direction along that axis. In the form (Axis0Left, AxisNLeft...,
      // Axis0Right, AxisNRight) where left and right refer to the direction
      // along the axis to add zeros to.
      const std::size_t num_pads = padding.size();
      for (std::size_t padIndex = 0; padIndex < num_pads; ++padIndex) {
        padding.push_back(padding[padIndex]);
      }

      std::vector<std::int64_t> dilation =
          HandleList<std::int64_t>(node->input(5)->node());
      // torch::jit::Value* output_padding = node->input(8);
      std::int64_t groups =
          *HandleConstant<std::int64_t>(node->input(8)->node());

      if (transposed && *transposed == 0) {
        // Create a "normal" convolution.
        newNode = poptorch::Create_conv(graph, inputs, dilation, groups, {},
                                        padding, stride);

      } else {
        logging::err("Transposed convolutions are not currently supported.");

        /* TODO(T22979) Re-enable once PopART supports transposed convolutions.
        // Output shape.
        // Give popart the shape of the output so it can autogenerate pads.
        std::vector<std::int64_t> outputShape = ShapeFromTensor(node->output())

        newNode = poptorch::Create_convtranspose(graph, inputs, dilation,
        groups, {}, {}, outputShape, padding, stride);
        */
      }
    } else if (kind == c10::aten::conv2d) {
      /*
      aten::conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride,
      int[] padding, int[] dilation, int groups) -> Tensor
      */
      auto input = node->input(0);
      auto kernel = node->input(1);

      std::vector<torch::jit::Value *> inputs{input, kernel};

      // Add bias if present.
      if (!IsNone(node->input(2)->node())) {
        inputs.push_back(node->input(2));
      }

      std::vector<std::int64_t> stride =
          HandleList<std::int64_t>(node->input(3)->node());
      std::vector<std::int64_t> padding =
          HandleList<std::int64_t>(node->input(4)->node());

      // Pytorch gives the padding as being the amount to pad in both
      // directions. Popart two arguments for each axis, the amount to pad in
      // each direction along that axis. In the form (Axis0Left, AxisNLeft...,
      // Axis0Right, AxisNRight) where left and right refer to the direction
      // along the axis to add zeros to.
      const std::size_t num_pads = padding.size();
      for (std::size_t padIndex = 0; padIndex < num_pads; ++padIndex) {
        padding.push_back(padding[padIndex]);
      }

      std::vector<std::int64_t> dilation =
          HandleList<std::int64_t>(node->input(5)->node());
      std::int64_t groups =
          *HandleConstant<std::int64_t>(node->input(6)->node());

      newNode = poptorch::Create_conv(graph, inputs, dilation, groups, {},
                                      padding, stride);
    } else if (kind == c10::aten::batch_norm) {
      // clang-format off
      /*
      aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor?
      running_mean, Tensor?  , bool training, float momentum, float
      eps, bool cudnn_enabled) -> Tensor
      */
      // clang-format on

      // Pytorch supports BatchNorm1D/2D/3D. PopART only supports 2D so we need
      // to reshape input into a 4D tensor.

      // Keep track of the original shape so we can convert back if we are
      // running BatchNorm1D or 3D.
      std::vector<std::int64_t> originalShape = ShapeFromTensor(node->input(0));

      // New 4D shape to perform the operation with.
      std::vector<std::int64_t> newShape = originalShape;

      // Turn the shape into a 4D tensor.
      if (originalShape.size() == 2) {
        // Add two singletons to pad to 4D.
        newShape.push_back(1);
        newShape.push_back(1);
      } else if (originalShape.size() == 3) {
        // Add one singleton to get to 4D.
        newShape.push_back(1);
      } else if (originalShape.size() == 5) {
        // Flatten last two dimensions to reduce to 4.
        newShape[3] *= newShape[4];
        newShape.pop_back();
      }

      // Input is value at 0th position.
      torch::jit::Value *input = node->input(0);

      // Reshape to 4D if needed.
      if (originalShape.size() != 4) {
        torch::jit::Node *reshape_in = CreateReshape(graph, input, newShape);
        input = reshape_in->output();
      }

      torch::jit::Value *weight = node->input(1);
      torch::jit::Value *bias = node->input(2);
      torch::jit::Value *running_mean = node->input(3);
      torch::jit::Value *running_var = node->input(4);

      // TODO(T22645): These will have to be checked if they are actual tensors
      // in the future.
      std::vector<torch::jit::Value *> inputTensors{input, weight, bias,
                                                    running_mean, running_var};

      float momentum = *HandleConstant<float>(node->input(6)->node());
      float epsilon = *HandleConstant<float>(node->input(7)->node());

      newNode = poptorch::Create_batchnormalization(graph, inputTensors, 1,
                                                    epsilon, momentum);

      // If we reshaped, reshape back.
      if (originalShape.size() != 4) {
        // Add the batch norm.

        // This is now the new node.
        newNode = CreateReshape(graph, newNode->output(), originalShape);
      }

    } else if (kind == c10::aten::max_pool1d || kind == c10::aten::avg_pool1d ||
               kind == c10::aten::max_pool2d || kind == c10::aten::avg_pool2d ||
               kind == c10::aten::max_pool3d || kind == c10::aten::avg_pool3d) {
      // clang-format off
      /*
        aten::max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[]
        padding, int[] dilation, bool ceil_mode) -> Tensor

        aten::avg_pool2d(Tensor self, int[] kernel_size, int[] stride, int[]
                         padding, bool ceil_mode, bool count_include_pad,
                         int? divisor_override) -> Tensor
     */
      // clang-format on
      std::vector<std::int64_t> kernel_size =
          HandleList<std::int64_t>(node->input(1)->node());
      std::vector<std::int64_t> stride =
          HandleList<std::int64_t>(node->input(2)->node());
      std::vector<std::int64_t> padding =
          HandleList<std::int64_t>(node->input(3)->node());

      // Pytorch gives the padding as being the amount to pad in both
      // directions. Popart two arguments for each axis, the amount to pad in
      // each direction along that axis. In the form (Axis0Left, AxisNLeft...,
      // Axis0Right, AxisNRight) where left and right refer to the direction
      // along the axis to add zeros to.
      const std::size_t num_pads = padding.size();
      for (std::size_t padIndex = 0; padIndex < num_pads; ++padIndex) {
        padding.push_back(padding[padIndex]);
      }

      if (kind == c10::aten::max_pool1d || kind == c10::aten::max_pool2d ||
          kind == c10::aten::max_pool3d) {
        newNode = poptorch::Create_maxpool(graph, {node->input(0)}, 1,
                                           kernel_size, padding, 0, stride);
      } else {
        // ceil_mode, countIncludePad, divisor_override are ignored for now due
        // to not being supported directly in popart.
        newNode = poptorch::Create_averagepool(graph, {node->input(0)},
                                               kernel_size, 0, padding, stride);
      }
    } else if (kind == c10::aten::adaptive_avg_pool2d ||
               kind == c10::aten::adaptive_max_pool2d) { // clang-format off
      // aten::adaptive_avg_pool2d(Tensor self, int[] output_size) -> Tensor
      // aten::adaptive_max_pool2d(Tensor self, int[] output_size) -> Tensor
      // clang-format on
      std::vector<std::int64_t> outputShape =
          HandleList<std::int64_t>(node->input(1)->node());

      c10::TensorTypePtr asTensor =
          node->input(0)->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();
      std::vector<std::int64_t> inputShape{*dims[2], *dims[3]};

      // Need to clean this code up.
      // TODO(tbd)
      const std::vector<int64_t> &stride{inputShape[0] / outputShape[0],
                                         inputShape[1] / outputShape[1]};

      const std::vector<int64_t> &kernel_shape{
          inputShape[0] - (outputShape[0] - 1) * stride[0],
          inputShape[1] - (outputShape[1] - 1) * stride[1]};
      const std::vector<int64_t> &padding{0, 0, 0, 0};

      if (kind == c10::aten::adaptive_avg_pool2d) {
        newNode = Create_averagepool(graph, {node->input(0)}, kernel_shape, 0,
                                     padding, stride);
      } else {
        logging::err("Adaptive max pooling isn't currently supported.");
        /* // TODO(T22978) Fix the number of inputs in PopParse so this can
           return 2.
           // Supported by Onnx.

            newNode = poptorch::Create_maxpool(graph,
           {node->input(0)}, 2, kernel_shape, padding, 0, stride);*/
      }
    } else if (kind == c10::aten::softmax) {
      // "aten::softmax(Tensor self, int dim, int? dtype) -> Tensor"

      std::int64_t dim = *HandleConstant<std::int64_t>(node->input(1)->node());

      c10::TensorTypePtr asTensor =
          node->input(0)->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();

      if (dim < 0) {
        dim = *dims.size() + dim;
      }

      newNode = Create_softmax(graph, {node->input(0)}, dim);
    } else if (kind == c10::aten::log10) {
      // Log10(X) = Log(X) / Log(10)

      // Add log(x)
      torch::jit::Node *logx = Create_log(graph, {node->inputs()[0]});

      // Add log10
      const double log10Const =
          2.302585092994045684017991454684364207601101488628772976033;
      torch::jit::Node *log10 = Create_ConstantFloat(graph, {log10Const}, {});

      // Add the divide.
      newNode = Create_div(graph, {logx->output(), log10->output()});
    } else if (kind == c10::aten::log1p) {
      // Log1p(x) = log(x + 1)

      // Add the one constant
      torch::jit::Node *one = Create_ConstantFloat(graph, {1.0}, {});

      // Add x + 1
      torch::jit::Node *add =
          Create_add(graph, {node->inputs()[0], one->output()});

      // Add the log
      newNode = Create_log(graph, {add->output()});
    } else if (kind == c10::aten::log2) {
      // Log2(X) = Log(X) / Log(2)

      // Add log(x)
      torch::jit::Node *logx = Create_log(graph, {node->inputs()[0]});

      // Add log2
      const double log2Const =
          0.693147180559945309417232121458176568075500134360255254120;
      torch::jit::Node *log2 = Create_ConstantFloat(graph, {log2Const}, {});

      // Add the divide.
      newNode = Create_div(graph, {logx->output(), log2->output()});
    } else if (kind == c10::aten::log_softmax) {
      // "aten::log_softmax(Tensor self, int dim, int? dtype) -> Tensor"

      std::int64_t dim = *HandleConstant<std::int64_t>(node->input(1)->node());

      c10::TensorTypePtr asTensor =
          node->input(0)->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();

      if (dim < 0) {
        dim = *dims.size() + dim;
      }

      newNode = Create_softmax(graph, {node->input(0)}, dim);

      newNode = Create_log(graph, {newNode->output()});
    } else if (kind == c10::aten::nll_loss) {
      // This is derived by me (stephenm@graphcore.ai) not parsed from the
      // pytorch headers like the others as I can't find it in them.

      // "aten::nll_loss(Tensor input, Tensor label, Tensor? weight, int
      // reduction, int ignore_index) -> Tensor"

      std::int64_t reduction =
          *HandleConstant<std::int64_t>(node->input(3)->node());
      std::int64_t ignore_index =
          *HandleConstant<std::int64_t>(node->input(4)->node());

      // Convert to popart reduce values.
      reduction = convertReduceToPopart(reduction);

      newNode = Create_nllloss(graph, {node->input(0), node->input(1)},
                               reduction, ignore_index);
    } else if (kind == c10::aten::l1_loss) {
      std::int64_t reduction =
          *HandleConstant<std::int64_t>(node->input(2)->node());

      // Convert to popart reduce values.
      reduction = convertReduceToPopart(reduction);

      // Popart calculates the L1 loss as being the difference from an input to
      // 0. So we have to manually subract the losses first.
      torch::jit::Node *subtract =
          Create_sub(graph, {node->input(0), node->input(1)});

      const float scale = 1.0f;
      newNode = Create_l1loss(graph, {subtract->output()}, scale, reduction);

    } else if (kind == c10::aten::mse_loss) {
      std::int64_t reduction =
          *HandleConstant<std::int64_t>(node->input(2)->node());

      // Convert to popart reduce values.
      reduction = convertReduceToPopart(reduction);

      // Subtract X - Y
      torch::jit::Node *subtract =
          Create_sub(graph, {node->input(0), node->input(1)});

      // Square it.
      torch::jit::Node *square =
          Create_mul(graph, {subtract->output(), subtract->output()});

      torch::jit::Node *finalNode = square;

      if (reduction == 0) {
        // Sum
        finalNode = Create_sum(graph, {square->output()});
      } else if (reduction == 1) {
        // Mean
        finalNode = Create_mean(graph, {square->output()});
      }

      newNode = Create_identityloss(graph, {finalNode->output()}, reduction);

    } else if (kind == Symbols::poptorch::begin_ipu_block) {
      // This could maybe be improved. Can we add attributes on the frontend?
      // TODO(tbd)
      newNode =
          graph.create(c10::Symbol::fromQualString("poptorch::begin_ipu_block"),
                       {}, node->outputs().size());
      graph.insertNode(newNode);

      // Convert the prim::Constant into an attribute.
      std::int64_t ipu_id =
          *HandleConstant<std::int64_t>(node->input()->node());
      newNode->i_(c10::Symbol::fromQualString("attr::ipu"), ipu_id);
    } else if (kind == c10::aten::select) {
      // clang-format off
      // aten::select(Tensor self, int dim, int index) -> Tensor

      // Note: there is also this overload which is not supported at the moment
      // aten::select(Tensor[] list, int idx) -> Tensor
      // clang-format on

      std::int64_t dim = *HandleConstant<std::int64_t>(node->input(1)->node());

      std::int64_t index =
          *HandleConstant<std::int64_t>(node->input(2)->node());

      newNode =
          Create_slice(graph, {node->input(0)}, {index + 1}, {index}, {dim});
    } else if (kind == c10::aten::slice) {
      // clang-format off
      // aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor // NOLINT
      // clang-format on

      std::int64_t dim = *HandleConstant<std::int64_t>(node->input(1)->node());

      std::int64_t start =
          *HandleConstant<std::int64_t>(node->input(2)->node());

      std::int64_t end = *HandleConstant<std::int64_t>(node->input(3)->node());
      if (end == 9223372036854775807 || end == -1) {
        c10::TensorTypePtr asTensor =
            node->input(0)->type()->cast<c10::TensorType>();
        c10::VaryingShape dims = asTensor->sizes();

        end = *dims[dim];
      }

      newNode = Create_slice(graph, {node->input(0)}, {end}, {start}, {dim});
    } else if (kind == c10::aten::permute) {
      // clang-format off
      // aten::permute(Tensor self, int[] dims) -> Tensor
      // clang-format on

      std::vector<std::int64_t> permutation =
          HandleList<std::int64_t>(node->input(1)->node());

      c10::TensorTypePtr asTensor =
          node->input(0)->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();

      std::for_each(permutation.begin(), permutation.end(),
                    [&](std::int64_t &val) {
                      if (val < 0) {
                        val = *dims.size() + val;
                      }
                    });

      newNode = Create_transpose(graph, {node->input(0)}, permutation);
    } else if (kind == c10::aten::contiguous) {
      // clang-format off
      // aten::contiguous(Tensor self, *, MemoryFormat memory_format=contiguous_format) -> Tensor // NOLINT
      // Returns a copy of the tensor but in contiguous memory.
      // clang-format on

      node->output()->replaceAllUsesWith(node->input(0));
      toDelete.insert(node);
    } else if (kind == c10::aten::transpose) {
      // clang-format off
      // aten::transpose(Tensor self, int dim0, int dim1) -> Tensor
      // clang-format on
      std::int64_t dim0 = *HandleConstant<std::int64_t>(node->input(1)->node());

      std::int64_t dim1 = *HandleConstant<std::int64_t>(node->input(2)->node());

      c10::TensorTypePtr asTensor =
          node->input(0)->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();

      // Convert that IR type into a C++ vector of ints. In popart the
      // permutation includes all elements (rotate last two elements with [0, 1,
      // 3, 2]) whereas in pytorch you only need to specify the dimensions being
      // moved (same operation, [3, 2]). So we need to make sure the IR reflects
      // that.
      std::vector<std::int64_t> permutation;
      for (std::uint64_t i = 0; i < *dims.size(); ++i) {
        permutation.push_back(i);
      }

      // Allow for python array style access.
      if (dim0 < 0) {
        dim0 = *dims.size() + dim0;
      }

      if (dim1 < 0) {
        dim1 = *dims.size() + dim1;
      }

      permutation[dim0] = dim1;
      permutation[dim1] = dim0;

      newNode = Create_transpose(graph, {node->input(0)}, permutation);
    } else if (kind == c10::aten::embedding) {
      // aten::embedding(Tensor weight, Tensor indices, int padding_idx, bool
      // scale_grad_by_freq, bool sparse) -> Tensor

      bool scale_grad_by_freq = *HandleConstant<bool>(node->input(3)->node());
      bool sparse = *HandleConstant<bool>(node->input(4)->node());

      if (scale_grad_by_freq || sparse) {
        std::cout << "Unsupported aten::embedding operation" << std::endl;
        newNode->dump();
        exit(0);
      }

      newNode = Create_gather(graph, {node->input(0), node->input(1)}, 0);
    } else if (kind == c10::aten::ones) {
      // clang-format off
      // aten::ones(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor // NOLINT
      // clang-format on
      c10::TensorTypePtr asTensor =
          node->outputs()[0]->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();
      std::vector<std::int64_t> operationShape;

      for (auto optionalInt : *dims.sizes()) {
        operationShape.push_back(*optionalInt);
      }

      switch (*asTensor->scalarType()) {
      case c10::ScalarType::Int:
      case c10::ScalarType::Long: {
        newNode = Create_ConstantInt(graph, {1}, operationShape);
        break;
      }
      case c10::ScalarType::Float: {
        newNode = Create_ConstantFloat(graph, {1.0}, operationShape);
        break;
      }
      default:
        ERROR("aten::ones of type " << c10::toString(*asTensor->scalarType())
                                    << " not supported");
      }
    } else if (kind == c10::aten::zeros) {
      // clang-format off
      // aten::zeros(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor // NOLINT
      // clang-format on
      c10::TensorTypePtr asTensor =
          node->outputs()[0]->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();
      std::vector<std::int64_t> operationShape;

      for (auto optionalInt : *dims.sizes()) {
        operationShape.push_back(*optionalInt);
      }

      switch (*asTensor->scalarType()) {
      case c10::ScalarType::Int:
      case c10::ScalarType::Long: {
        newNode = Create_ConstantInt(graph, {0}, operationShape);
        break;
      }
      case c10::ScalarType::Float: {
        newNode = Create_ConstantFloat(graph, {0.0}, operationShape);
        break;
      }
      default:
        ERROR("aten::zeros of type " << c10::toString(*asTensor->scalarType())
                                     << " not supported");
      }
    } else if (kind == c10::aten::to) {
      // clang-format off
      // aten::to(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)" // NOLINT
      // aten::to(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)" // NOLINT
      // aten::to(Tensor(a) self, bool non_blocking=False, bool copy=False) -> Tensor(a|b)" // NOLINT
      // clang-format on

      c10::TensorTypePtr outputType =
          node->outputs()[0]->type()->expect<c10::TensorType>();

      c10::TensorTypePtr inputType =
          node->input(0)->type()->expect<c10::TensorType>();

      c10::ScalarType outAsScalar = *outputType->scalarType();
      c10::ScalarType inAsScalar = *inputType->scalarType();

      // Remove the node if casting to the same type.
      if (outAsScalar == inAsScalar) {
        node->output()->replaceAllUsesWith(node->input(0));
        toDelete.insert(node);
        continue;
      }

      // Otherwise cast as normal.
      newNode = Create_Cast(graph, node->input(0), outAsScalar);

    } else if (kind == c10::aten::rsub) {
      // clang-format off
      // Tensor aten::rsub(const Tensor& self, const Tensor& other, Scalar alpha) // NOLINT
      // clang-format on
      // We are ignoring alpha here.

      torch::jit::Value *other = node->input(1);

      std::optional<float> asScalar = HandleConstant<float>(other->node());

      // This operation can also take a scalar for other. If it is that overload
      // then we have to add it as a popart scalar and work with that instead.
      if (asScalar) {
        torch::jit::Node *asConstant =
            Create_ConstantFloat(graph, {*asScalar}, {1});

        other->replaceAllUsesWith(asConstant->output());

        // Mark it for deletion.
        toDelete.insert(other->node());

        // Use the popart constant instead.
        other = asConstant->output();
      }

      newNode = Create_sub(graph, {other, node->input(0)});
    } else if (kind == c10::aten::arange) {
      // clang-format off
      // aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory) // NOLINT
      // aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory) // NOLINT
      // clang-format on

      if (node->inputs().size() != 5) {
        std::cerr << "Unsupported arrange op";
        newNode->dump();
      }

      std::vector<std::int64_t> vals;
      std::size_t end = *HandleConstant<std::int64_t>(node->input(0)->node());
      for (std::size_t start = 0; start < end; ++start) {
        vals.push_back(start);
      }

      newNode = Create_ConstantInt(graph, vals,
                                   {static_cast<std::int64_t>(vals.size())});
    } else if (kind == Symbols::poptorch::identity_loss) {
      std::int64_t reduction =
          *HandleConstant<std::int64_t>(node->input(1)->node());

      newNode = Create_identityloss(graph, {node->input(0)}, reduction);
    } else if (kind == c10::aten::layer_norm) {
      // clang-format off
      // aten::layer_norm(Tensor input,int[] normalized_shape, Tensor? weight,
      //                Tensor? bias, float eps, bool cudnn_enable) -> Tensor
      // clang-format on

      // Tensor to normalise.
      torch::jit::Value *X = node->input(0);

      // Bias to add
      torch::jit::Value *gamma = node->input(2);

      // Weight to multiply.
      torch::jit::Value *beta = node->input(3);

      const float epsilon = *HandleConstant<float>(node->input(4)->node());

      // Pytorch normalizes across arbitrary number of dimensions from the end.
      // We flatten into a [M, N] array and normalize the N.
      std::vector<std::int64_t> normalizedShape =
          HandleList<int64_t>(node->input(1)->node());
      const std::int64_t axis = -normalizedShape.size();

      // Flatten into [M, N]
      torch::jit::Node *flatten = Create_flatten(graph, {X}, axis);

      // Normalize.
      torch::jit::Node *normalize = Create_groupnormalization(
          graph, {flatten->output(), gamma, beta}, 1, epsilon);

      // Reshape back into the expected shape.
      c10::TensorTypePtr convertedToTensor =
          node->output()->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = convertedToTensor->sizes();

      std::vector<std::int64_t> originalShape;

      for (auto optionalInt : *dims.sizes()) {
        originalShape.push_back(*optionalInt);
      }

      // Perform the reshape.
      newNode = CreateReshape(graph, normalize->output(), originalShape);

    } else if (kind == Symbols::poptorch::identity_loss) {
      std::int64_t reduction =
          *HandleConstant<std::int64_t>(node->input(1)->node());

      newNode = Create_identityloss(graph, {node->input(0)}, reduction);
    } else if (kind == c10::aten::split || kind == c10::aten::chunk) {
      // clang-format off
      // aten::split(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]"
      // aten::split(Tensor self, int split_sizes, int dim=0) -> Tensor[]"
      // aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]
      // clang-format on

      // Get the shape of the input.
      c10::TensorTypePtr asTensor =
          node->input(0)->type()->expect<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();

      // Pythonic axis translation.
      const std::int64_t dim =
          *HandleConstant<std::int64_t>(node->input(2)->node());
      const std::int64_t axis = dim >= 0 ? dim : *dims.size() + dim;

      // Size of each split ignoring the remainder at the end.
      std::vector<std::int64_t> sizeOfEachSplit;

      // Split size can either be the number of splits or the size of the
      // splits.
      std::optional<std::int64_t> splitSize =
          HandleConstant<std::int64_t>(node->input(1)->node());

      if (kind == c10::aten::chunk) {
        // Chunk takes in the *number of chunks*. Canonicalise it to *size of
        // chunks*.
        ERROR_ON_MSG(
            !splitSize,
            "Aten chunk node does not have a integer number of chunks!");
        std::int64_t sliceSize = *dims[axis] / *splitSize;
        for (int i = 0; i < *splitSize; ++i) {
          sizeOfEachSplit.push_back(sliceSize);
        }

        // Add an extra slice for the remainder.
        if (*dims[axis] % *splitSize != 0) {
          sizeOfEachSplit.push_back(*dims[axis] % *splitSize);
        }
      } else if (splitSize) {
        // Split takes in the size of each chunk.
        std::int64_t sliceSize = *splitSize;
        for (int i = 0; i < *dims[axis] / sliceSize; ++i) {
          sizeOfEachSplit.push_back(sliceSize);
        }

        // Add an extra slice for the remainder.
        if (*dims[axis] % *splitSize != 0) {
          sizeOfEachSplit.push_back(*dims[axis] % *splitSize);
        }
      } else {
        sizeOfEachSplit = HandleList<std::int64_t>(node->input(1)->node());
      }

      // Rolling index to track where we are in the tensor.
      std::int64_t index = 0;

      // The result of each slice.
      std::vector<torch::jit::Value *> slices;

      // Slice up according to the canonicalised split vector.
      for (std::int64_t sliceSize : sizeOfEachSplit) {
        // Create a slice.
        newNode = Create_slice(graph, {node->input(0)}, {index + sliceSize},
                               {index}, {axis});

        // Add the slice to the graph.
        slices.push_back(newNode->output());

        // Move along in the vector dimension.
        index += sliceSize;
      }

      newNode = graph.create(at::prim::ListConstruct, slices);
      graph.insertNode(newNode);
    } else if (kind == c10::aten::masked_fill) {
      // clang-format off
      // Derived from documentation
      // aten::masked_fill(Tensor self, Tensor mask, Tensor other) -> Tensor
      // clang-format on

      // Apply by performing the following operation
      // inverseMask = -(mask - 1)
      // self * inverseMask + mask * other

      // Cast the mask to int32.
      torch::jit::Node *mask = Create_Cast(graph, node->input(1), c10::kInt);

      // Create an inverse mask via -(mask - 1)
      torch::jit::Node *negativeOne = Create_ConstantInt(graph, {-1}, {1});

      torch::jit::Node *inverseMask =
          Create_add(graph, {mask->output(), negativeOne->output()});

      inverseMask = Create_neg(graph, {inverseMask->output()});

      // Prepare input and update
      mask = Create_Cast(graph, node->input(1), c10::kFloat);

      float otherAsConst = *HandleConstant<float>(node->input(2)->node());
      torch::jit::Node *other =
          Create_ConstantFloat(graph, {otherAsConst}, {1});

      torch::jit::Node *update =
          Create_mul(graph, {mask->output(), other->output()});

      // Create holes in the original so we can add into it.
      inverseMask = Create_Cast(graph, inverseMask->output(), c10::kFloat);

      torch::jit::Node *self =
          Create_mul(graph, {node->input(0), inverseMask->output()});

      newNode = Create_add(graph, {self->output(), update->output()});
    } else if (kind == c10::aten::rsqrt) {
      // rsqrt =  1 / sqrt(x)
      torch::jit::Node *sqrt = Create_sqrt(graph, {node->input()});

      newNode = Create_reciprocal(graph, {sqrt->output()});
    } else if (kind == c10::aten::expm1) {
      // expm1 = exp(x) - 1

      // exp(x)
      torch::jit::Node *exp = Create_exp(graph, {node->input()});

      // Add the one constant
      torch::jit::Node *one = Create_ConstantFloat(graph, {1.0}, {});

      newNode = Create_sub(graph, {exp->output(), one->output()});
    } else if (kind == c10::aten::trunc) {
      // Drop the exponent by casting to int and back.
      torch::jit::Node *toInt = Create_Cast(graph, node->input(), c10::kInt);

      newNode = Create_Cast(graph, toInt->output(), c10::kFloat);

    } else if (kind == c10::aten::frac) {
      // Frac(x) = x - trunc(x)

      // Drop the exponent by casting to int and back.
      torch::jit::Node *toInt = Create_Cast(graph, node->input(), c10::kInt);

      torch::jit::Node *trunc =
          Create_Cast(graph, toInt->output(), c10::kFloat);

      newNode = Create_sub(graph, {node->input(), trunc->output()});
    } else if (kind == c10::aten::round) {
      // round(x) = trunc(x + sign(x)*0.5)

      // Add 0.5 as constant.
      torch::jit::Node *zeroPointFive = Create_ConstantFloat(graph, {0.5}, {});

      torch::jit::Node *sign = Create_sign(graph, {node->input()});

      torch::jit::Node *broadcastBySign =
          Create_mul(graph, {sign->output(), zeroPointFive->output()});

      torch::jit::Node *addition =
          Create_add(graph, {node->input(), broadcastBySign->output()});

      // Drop the exponent by casting to int and back.
      torch::jit::Node *toInt =
          Create_Cast(graph, addition->output(), c10::kInt);

      newNode = Create_Cast(graph, toInt->output(), c10::kFloat);
    } else if (kind == c10::aten::floor_divide) {
      // aten::floor_divide(Tensor x, Tensor y) -> Tensor
      // floor_divide(x, y) = floor(x)/floor(y)

      torch::jit::Node *x = Create_floor(graph, {node->inputs()[0]});
      torch::jit::Node *y = Create_floor(graph, {node->inputs()[1]});

      newNode = Create_div(graph, {x->output(), y->output()});

    } else if (kind == c10::aten::true_divide) {
      // aten::true_divide(Tensor x, Tensor y) -> Tensor
      // true_divide(x, y) = (float)x / (float)y

      torch::jit::Node *x = Create_Cast(graph, node->inputs()[0], c10::kFloat);

      torch::jit::Node *y = Create_Cast(graph, node->inputs()[1], c10::kFloat);

      newNode = Create_div(graph, {x->output(), y->output()});
    } else if (kind == c10::aten::argmax || kind == c10::aten::argmin) {
      // clang-format off
      //  aten::argmin(Tensor in, int? dim, int keep_dims) -> Tensor
      //  aten::argmax(Tensor in, int? dim, int keep_dims) -> Tensor
      // dim (int) – the dimension to reduce. If None, the argmax
      //             of the flattened input is returned.
      // clang-format on

      torch::jit::Value *input = node->input(0);
      std::optional<std::int64_t> dim =
          HandleConstant<std::int64_t>(node->inputs()[1]->node());
      std::int64_t keepDim =
          *HandleConstant<std::int64_t>(node->inputs()[2]->node());

      // If dim is not provided we will flatten input so just use 0 in that
      // case.
      std::int64_t dimToUse = 1;

      // Check if dim is NONE.
      if (!dim) {
        torch::jit::Node *flatten =
            Create_flatten(graph, {node->inputs()[0]}, 0);
        input = flatten->output();
      } else {
        dimToUse = *dim;
      }

      // Create the actual argmax/argmin.
      if (kind == c10::aten::argmax) {
        newNode = Create_argmax(graph, {input}, dimToUse, keepDim);
      } else {
        newNode = Create_argmin(graph, {input}, dimToUse, keepDim);
      }

    } else if (kind == c10::aten::prod || kind == c10::aten::mean ||
               kind == c10::aten::sum || kind == c10::aten::logsumexp) {
      // clang-format off

      // Reductions have two overloads. The first is:
      // aten::mean(Tensor self, int[] dim, int keepdim, Tensor? out)) -> tensor

      // The second is:
      // aten::mean(Tensor self, int? dtype)) -> tensor

      // clang-format on

      torch::jit::Value *input = node->input(0);

      std::vector<std::int64_t> axes{};
      std::int64_t keepdim = 0;

      // Case 2.
      if (node->inputs().size() == 2) {
        torch::jit::Node *flatten =
            Create_flatten(graph, {node->inputs()[0]}, 0);
        input = flatten->output();
        axes = {1};
      } else {
        // Case 1.
        // Sometimes the dimensions are just one int.
        std::optional<std::int64_t> asInt =
            HandleConstant<std::int64_t>(node->inputs()[1]->node());

        if (asInt) {
          axes.push_back(*asInt);
        } else {
          axes = HandleListConstruct<std::int64_t>(node->inputs()[1]->node());
        }

        keepdim = *HandleConstant<std::int64_t>(node->inputs()[2]->node());
      }

      // Output the correct reduction.
      if (kind == c10::aten::prod) {
        newNode = Create_reduceprod(graph, {input}, axes, keepdim);
      } else if (kind == c10::aten::mean) {
        newNode = Create_reducemean(graph, {input}, axes, keepdim);
      } else if (kind == c10::aten::sum) {
        newNode = Create_reducesum(graph, {input}, axes, keepdim);
      } else if (kind == c10::aten::logsumexp) {
        newNode = Create_reducelogsumexp(graph, {input}, axes, keepdim);
      } else {
        ERROR("Popart Canonicalisation: UNREACHABLE reached in reductions.");
      }
    } else if (kind == c10::aten::binary_cross_entropy) {
      // clang-format off
      // aten::binary_cross_entropy(Tensor input, Tensor target,
      //                            Tensor? weight, int reduction)
      // clang-format on

      // L = loss, w = weight, y= target, x = input.
      // Algorithm is: L = - w * (y *log(x) + (1 - y)*log(1 - x))

      // The input.
      torch::jit::Value *x = node->input(0);

      // The target.
      torch::jit::Value *y = node->input(1);

      // Optional weight term.
      torch::jit::Value *weight = node->input(2);

      // Loss reduction.
      std::int64_t reduction =
          *HandleConstant<std::int64_t>(node->input(3)->node());

      // Convert to popart reduce values.
      reduction = convertReduceToPopart(reduction);

      // Add the one constant
      torch::jit::Node *one = Create_ConstantFloat(graph, {1.0}, {});

      torch::jit::Node *logX = Create_log(graph, {x});

      // Log(x)*y
      torch::jit::Node *logXMulY = Create_mul(graph, {y, logX->output()});

      // Do (1 - y) and (1 - x)
      torch::jit::Node *xMinusOne = Create_sub(graph, {one->output(), x});
      torch::jit::Node *yMinusOne = Create_sub(graph, {one->output(), y});

      // Log(1 - x)
      torch::jit::Node *logXMinusOne = Create_log(graph, {xMinusOne->output()});

      // (1 -y)*Log(1 - x)
      torch::jit::Node *subsMultiplied =
          Create_mul(graph, {yMinusOne->output(), logXMinusOne->output()});

      // Log(x)*y + (1 -y)*Log(1 - x)
      torch::jit::Node *addTerms =
          Create_add(graph, {logXMulY->output(), subsMultiplied->output()});

      torch::jit::Node *finalNode = addTerms;

      if (weight->node()->kind() != c10::prim::Constant) {
        finalNode = Create_mul(graph, {addTerms->output(), weight});
      }

      finalNode = Create_neg(graph, {addTerms->output()});
      if (reduction == 0) {
        // Sum
        finalNode = Create_sum(graph, {finalNode->output()});
      } else if (reduction == 1) {
        // Mean
        finalNode = Create_mean(graph, {finalNode->output()});
      }

      newNode = Create_identityloss(graph, {finalNode->output()}, reduction);

    } else if (kind == c10::aten::lstm) {
      // clang-format off
      // aten::lstm(Tensor self, Tensor[] hx, Tensor[] weights, bool bias,
      // int num_layers, float dropout, bool training, bool bidirectional,
      // bool batch_first) -> Tensor, (Tensor, Tensor)
      // clang-format on

      torch::jit::Value *input = node->input(0);

      torch::jit::ArrayRef<torch::jit::Value *> hiddenLayers =
          node->input(1)->node()->inputs();
      torch::jit::ArrayRef<torch::jit::Value *> weightsList =
          node->input(2)->node()->inputs();

      bool useBias = *HandleConstant<bool>(node->input(3)->node());
      ERROR_ON_MSG(!useBias, "LSTM without biases not supported");
      std::int64_t numLayers =
          *HandleConstant<std::int64_t>(node->input(4)->node());
      ERROR_ON_MSG(numLayers != 1, "Only LSTM with 1 layer supported");

      float dropout = *HandleConstant<float>(node->input(5)->node());
      ERROR_ON_MSG(dropout != 0.0f, "LSTM only supports dropout = 0.0");

      bool bidirectional = *HandleConstant<bool>(node->input(7)->node());
      ERROR_ON_MSG(bidirectional, "bidirectional LSTM not supported");

      bool batchFirst = *HandleConstant<bool>(node->input(8)->node());

      // An LSTM state is made of 4 values
      constexpr std::uint64_t stateSize = 4;
      const std::int64_t numWeights =
          *weightsList[0]->type()->cast<c10::TensorType>()->sizes()[0];
      ERROR_ON(numWeights % stateSize != 0);
      const std::int64_t numHiddenLayers = numWeights / stateSize;

      // def reshape_weights(onnx_weights):
      //    ws = builder.aiOnnx.split([w], 4, 1, [hidden_size] * 4)
      //    ws = [builder.aiOnnx.transpose([i], [0, 2, 1]) for i in ws]
      //    ws = builder.aiOnnx.concat([ws[i] for i in (2, 0, 3, 1)], 0)
      //    return ws
      //
      // Note: onnx weights are in IOFC order while Torch uses IFCO
      //
      // Biases don't need to be transposed
      auto reshapeTensor = [&](torch::jit::Value *values, bool areWeights) {
        const std::uint64_t numDimsWithoutBatch = areWeights ? 2 : 1;
        std::vector<std::int64_t> shape = ShapeFromTensor(values);
        if (shape.size() == numDimsWithoutBatch) {
          // Add a batch dimension
          shape.insert(shape.begin(), 1);
          torch::jit::Node *reshape = CreateReshape(graph, values, shape);
          values = reshape->output();
        }
        torch::jit::Node *states =
            Create_split(graph, {values}, stateSize, 1,
                         {numHiddenLayers, numHiddenLayers, numHiddenLayers,
                          numHiddenLayers});
        std::vector<torch::jit::Value *> slices;
        for (std::uint64_t i = 0; i < stateSize; ++i) {
          if (areWeights) {
            // Weights also need to be transposed
            torch::jit::Node *transposed =
                Create_transpose(graph, {states->output(i)}, {0, 2, 1});
            slices.push_back(transposed->output());
          } else {
            slices.push_back(states->output(i));
          }
        }
        torch::jit::Node *concat = Create_concat(
            graph, {slices[1], slices[0], slices[2], slices[3]}, 0);
        return concat->output();
      };

      torch::jit::Node *concatWeights =
          Create_concat(graph,
                        {reshapeTensor(weightsList[0], true),
                         reshapeTensor(weightsList[1], true)},
                        1);
      torch::jit::Node *combineBiases =
          Create_addNotInPlace(graph, reshapeTensor(weightsList[2], false),
                               reshapeTensor(weightsList[3], false));

      torch::jit::Node *concatStates =
          Create_concat(graph, {hiddenLayers[0], hiddenLayers[1]}, 0);

      // Transpose output BSF -> SBF
      if (batchFirst) {
        torch::jit::Node *transpose =
            Create_transpose(graph, {input}, {1, 0, 2});
        input = transpose->output();
      }
      std::vector<torch::jit::Value *> args;
      args.push_back(input);
      args.push_back(concatWeights->output()); // input weights + output_weights
      args.push_back(combineBiases->output()); // biases
      args.push_back(concatStates->output());  // init_states

      torch::jit::Node *lstm = Create_lstm(graph, args, 1);

      // Keep the last slice from Y
      torch::jit::Node *Y_h =
          Create_slice(graph, {lstm->output(0)}, {INT_MAX}, {-1}, {0});

      torch::jit::Value *output = lstm->output(0);
      // Transpose output SBF -> BSF
      if (batchFirst) {
        torch::jit::Node *transpose =
            Create_transpose(graph, {output}, {1, 0, 2});
        output = transpose->output();
      }

      ERROR_ON(node->outputs().size() != 3);
      if (node->hasUses()) {
        ReplaceOutputUse(node->output(0), output);
        ReplaceOutputUse(node->output(1), Y_h->output());
        ReplaceOutputUse(node->output(2), lstm->output(1));
      }

      toDelete.insert(node);
    } else if (kind == c10::aten::ge || kind == c10::aten::le) {
      torch::jit::Node *comparison = nullptr;
      torch::jit::Value *lhs =
          HandleParamOrConstantNoCast(graph, node->input(0));
      torch::jit::Value *rhs =
          HandleParamOrConstantNoCast(graph, node->input(1));

      // Node will either be < or >.
      if (kind == c10::aten::ge) {
        comparison = Create_greater(graph, {lhs, rhs});
      } else {
        comparison = Create_less(graph, {lhs, rhs});
      }

      // We do a check for ==
      torch::jit::Node *equal = Create_equal(graph, {lhs, rhs});

      // The final node will be a combination of equals and less or greater.
      newNode =
          Create_logical_or(graph, {equal->output(), comparison->output()});
    } else if (kind == c10::aten::ne) {
      torch::jit::Value *lhs =
          HandleParamOrConstantNoCast(graph, node->input(0));
      torch::jit::Value *rhs =
          HandleParamOrConstantNoCast(graph, node->input(1));

      // Not(equal(lhs, rhs))
      torch::jit::Node *equal = Create_equal(graph, {lhs, rhs});
      newNode = Create_logical_not(graph, {equal->output()});
    }

    // If we have a new node add it and replace the old use.
    if (newNode) {
      // Mark this node for deletion.
      toDelete.insert(node);
      ERROR_ON(node->outputs().size() != newNode->outputs().size());

      if (node->hasUses()) {
        for (std::uint64_t i = 0; i < node->outputs().size(); ++i) {
          ReplaceOutputUse(node, newNode, i);
        }
      }
    }
  }

  // Remove any dead nodes.
  for (torch::jit::Node *node : toDelete) {
    SearchAndPossiblyDestroy(node);
  }
}

} // namespace

void Canonicalize(torch::jit::Graph &graph) {
  CanonicalizeImpl converter;
  converter.Run(graph);
}

} // namespace poptorch
