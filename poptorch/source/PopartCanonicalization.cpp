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

  template <typename T>
  torch::jit::Value *HandleParamOrConstant(torch::jit::Graph &graph,
                                           torch::jit::Value *operand);

  // Turn a parameter constant into an IR constant.
  torch::jit::Node *CreateIRConstant(torch::jit::Graph &graph,
                                     torch::jit::Value *node);

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
    } else {
      return std::nullopt;
    }
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
  std::size_t index = *HandleConstant<std::size_t>(node->inputs()[1]->node());

  // Get the shape of the tensor.
  c10::TensorTypePtr asTensor =
      node->inputs()[0]->type()->cast<c10::TensorType>();
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

template <typename T>
torch::jit::Value *
CanonicalizeImpl::HandleParamOrConstant(torch::jit::Graph &graph,
                                        torch::jit::Value *operand) {
  torch::jit::Value *valueToReturn = operand;
  torch::jit::Node *constant = CreateIRConstant(graph, operand);

  if (constant) {
    constant->insertBefore(operand->node());

    torch::jit::Node *cast = CastToType<T>(graph, constant->output());
    cast->insertAfter(constant);
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
    torch::jit::Node *newNode = nullptr;

    torch::jit::Symbol kind = node->kind();

    // We have a dummy if statement so we can daisy chain the rest of the "else
    // if's" off of it.
    if (kind == c10::aten::view || kind == c10::aten::unsqueeze ||
        kind == c10::aten::expand || kind == c10::aten::flatten ||
        kind == c10::aten::reshape || kind == c10::aten::squeeze) {
      // clang-format off

      // aten::view(Tensor self, int[] size) -> Tensor
      // aten::unsqueeze(Tensor self, int dim) -> Tensor
      // aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor
      // clang-format on

      // Extract the type from the pytorch IR.
      c10::TensorTypePtr asTensor =
          node->output()->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();

      // Convert that IR type into a C++ vector of ints.
      std::vector<std::int64_t> newShape;
      for (auto optionalInt : *dims.sizes()) {
        newShape.push_back(*optionalInt);
      }

      // Reshape the tensor into that shape.
      newNode = CreateReshape(graph, node->inputs()[0], newShape);
    } else if (kind == c10::aten::expand_as) {
      // clang-format off
      // aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor
      // aten::expand_as(Tensor self, Tensor other) -> Tensor
      // clang-format on

      // Extract the type from the pytorch IR.
      c10::TensorTypePtr selfTensor =
          node->inputs()[0]->type()->expect<c10::TensorType>();
      c10::VaryingShape selfDims = selfTensor->sizes();

      std::int64_t oldElemCount = 0;
      for (auto optionalInt : *selfDims.sizes()) {
        oldElemCount += *optionalInt;
      }

      // Extract the type from the pytorch IR.
      c10::TensorTypePtr asTensor =
          node->inputs()[1]->type()->expect<c10::TensorType>();
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
        newNode = CreateReshape(graph, node->inputs()[0], newShape);
      } else {
        newNode = Create_ConstantInt(graph, newShape,
                                     {static_cast<int64_t>(newShape.size())});
        newNode->insertBefore(node);

        newNode = Create_Cast(graph, newNode->output(), c10::kLong);
        newNode->insertBefore(node);

        newNode = Create_expand(graph, {node->inputs()[0], newNode->output()});
      }
    }
#define HANDLE(Index, Type) *HandleConstant<Type>(node->inputs()[Index]->node())
#define HANDLE_LIST(Index, Type)                                               \
  HandleListConstruct<Type>(node->inputs()[Index]->node())
#define HANDLE_TENSOR_LIST(Index)                                              \
  HandleTensorList(node->inputs()[Index]->node())
#define PARAM(Index) node->inputs()[Index]
#define COMMA ,
#define NONE

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
    alphaConst->insertAfter(alphaValue->node());                               \
    torch::jit::Node *alphaNode =                                              \
        Create_mul(graph, {alphaConst->output(), valueToMultiply});            \
    alphaNode->insertAfter(alphaConst);                                        \
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
          HandleConstant<std::int64_t>(node->inputs()[6]->node());

      torch::jit::Value *input = node->inputs()[0];
      torch::jit::Value *kernel = node->inputs()[1];

      std::vector<torch::jit::Value *> inputs{input, kernel};

      if (!IsNone(node->inputs()[2]->node())) {
        inputs.push_back(node->inputs()[2]);
      }

      std::vector<std::int64_t> stride =
          HandleList<int64_t>(node->inputs()[3]->node());
      std::vector<std::int64_t> padding =
          HandleList<std::int64_t>(node->inputs()[4]->node());

      // Slight workaround for current padding mechanism here.
      padding.push_back(padding[0]);
      padding.push_back(padding[1]);

      std::vector<std::int64_t> dilation =
          HandleList<std::int64_t>(node->inputs()[5]->node());
      // torch::jit::Value* output_padding = node->inputs()[8];
      std::int64_t groups =
          *HandleConstant<std::int64_t>(node->inputs()[8]->node());

      if (transposed && *transposed == 0) {
        // Create a "normal" convolution.
        newNode = poptorch::Create_conv(graph, inputs, dilation, groups, {},
                                        padding, stride);

      } else {
        logging::err("CURRENTLY UNSUPPORTED CONVOLUTION!!!\n{}", *newNode);
      }
    } else if (kind == c10::aten::conv2d) {
      /*
      aten::conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride,
      int[] padding, int[] dilation, int groups) -> Tensor
      */
      auto input = node->inputs()[0];
      auto kernel = node->inputs()[1];

      std::vector<torch::jit::Value *> inputs{input, kernel};

      // Add bias if present.
      if (!IsNone(node->inputs()[2]->node())) {
        inputs.push_back(node->inputs()[2]);
      }

      std::vector<std::int64_t> stride =
          HandleList<std::int64_t>(node->inputs()[3]->node());
      std::vector<std::int64_t> padding =
          HandleList<std::int64_t>(node->inputs()[4]->node());

      // Slight workaround for current padding mechanism here.
      padding.push_back(padding[0]);
      padding.push_back(padding[1]);

      std::vector<std::int64_t> dilation =
          HandleList<std::int64_t>(node->inputs()[5]->node());
      std::int64_t groups =
          *HandleConstant<std::int64_t>(node->inputs()[6]->node());

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

      torch::jit::Value *input = node->inputs()[0];
      torch::jit::Value *weight = node->inputs()[1];
      torch::jit::Value *bias = node->inputs()[2];
      torch::jit::Value *running_mean = node->inputs()[3];
      torch::jit::Value *running_var = node->inputs()[4];

      // TODO(T22645): These will have to be checked if they are actual tensors
      // in the future.
      std::vector<torch::jit::Value *> inputTensors{input, weight, bias,
                                                    running_mean, running_var};

      float momentum = *HandleConstant<float>(node->inputs()[6]->node());
      float epsilon = *HandleConstant<float>(node->inputs()[7]->node());

      newNode = poptorch::Create_batchnormalization(graph, inputTensors, 1,
                                                    epsilon, momentum);

    } else if (kind == c10::aten::max_pool2d || kind == c10::aten::avg_pool2d) {
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
          HandleList<std::int64_t>(node->inputs()[1]->node());
      std::vector<std::int64_t> stride =
          HandleList<std::int64_t>(node->inputs()[2]->node());
      std::vector<std::int64_t> padding =
          HandleList<std::int64_t>(node->inputs()[3]->node());

      // Slight workaround for current padding mechanism here.
      padding.push_back(padding[0]);
      padding.push_back(padding[1]);

      if (kind == c10::aten::max_pool2d) {
        newNode = poptorch::Create_maxpool(graph, {node->inputs()[0]}, 1,
                                           kernel_size, padding, 0, stride);
      } else {
        // ceil_mode, countIncludePad, divisor_override are ignored for now due
        // to not being supported directly in popart.
        newNode = poptorch::Create_averagepool(graph, {node->input(0)},
                                               kernel_size, 0, padding, stride);
      }
    } else if (kind == c10::aten::adaptive_avg_pool2d) {
      // clang-format off
      // aten::adaptive_avg_pool2d(Tensor self, int[] output_size) -> Tensor
      // clang-format on
      std::vector<std::int64_t> outputShape =
          HandleList<std::int64_t>(node->inputs()[1]->node());

      c10::TensorTypePtr asTensor =
          node->inputs()[0]->type()->cast<c10::TensorType>();
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

      newNode = Create_averagepool(graph, {node->inputs()[0]}, kernel_shape, 0,
                                   padding, stride);
    } else if (kind == c10::aten::softmax) {
      // "aten::softmax(Tensor self, int dim, int? dtype) -> Tensor"

      std::int64_t dim =
          *HandleConstant<std::int64_t>(node->inputs()[1]->node());

      c10::TensorTypePtr asTensor =
          node->inputs()[0]->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();

      if (dim < 0) {
        dim = *dims.size() + dim;
      }

      newNode = Create_softmax(graph, {node->inputs()[0]}, dim);
    } else if (kind == c10::aten::log10) {
      // Log10(X) = Log(X) / Log(10)

      // Add log(x)
      torch::jit::Node *logx = Create_log(graph, {node->inputs()[0]});
      logx->insertBefore(node);

      // Add log10
      const double log10Const =
          2.302585092994045684017991454684364207601101488628772976033;
      torch::jit::Node *log10 = Create_ConstantFloat(graph, {log10Const}, {});
      log10->insertBefore(node);

      // Add the divide.
      newNode = Create_div(graph, {logx->output(), log10->output()});
    } else if (kind == c10::aten::log1p) {
      // Log1p(x) = log(x + 1)

      // Add the one constant
      torch::jit::Node *one = Create_ConstantFloat(graph, {1.0}, {});
      one->insertBefore(node);

      // Add x + 1
      torch::jit::Node *add =
          Create_add(graph, {node->inputs()[0], one->output()});
      add->insertBefore(node);

      // Add the log
      newNode = Create_log(graph, {add->output()});
    } else if (kind == c10::aten::log2) {
      // Log2(X) = Log(X) / Log(2)

      // Add log(x)
      torch::jit::Node *logx = Create_log(graph, {node->inputs()[0]});
      logx->insertBefore(node);

      // Add log2
      const double log2Const =
          0.693147180559945309417232121458176568075500134360255254120;
      torch::jit::Node *log2 = Create_ConstantFloat(graph, {log2Const}, {});
      log2->insertBefore(node);

      // Add the divide.
      newNode = Create_div(graph, {logx->output(), log2->output()});
    } else if (kind == c10::aten::log_softmax) {
      // "aten::log_softmax(Tensor self, int dim, int? dtype) -> Tensor"

      std::int64_t dim =
          *HandleConstant<std::int64_t>(node->inputs()[1]->node());

      c10::TensorTypePtr asTensor =
          node->inputs()[0]->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();

      if (dim < 0) {
        dim = *dims.size() + dim;
      }

      newNode = Create_softmax(graph, {node->inputs()[0]}, dim);

      newNode->insertBefore(node);
      newNode = Create_log(graph, {newNode->output()});
    } else if (kind == c10::aten::nll_loss) {
      // This is derived by me (stephenm@graphcore.ai) not parsed from the
      // pytorch headers like the others as I can't find it in them.

      // "aten::nll_loss(Tensor input, Tensor label, Tensor? weight, int
      // reduction, int ignore_index) -> Tensor"

      std::int64_t reduction =
          *HandleConstant<std::int64_t>(node->inputs()[3]->node());
      std::int64_t ignore_index =
          *HandleConstant<std::int64_t>(node->inputs()[4]->node());

      // Convert to popart reduce values.
      reduction = convertReduceToPopart(reduction);

      newNode = Create_nllloss(graph, {node->inputs()[0], node->inputs()[1]},
                               reduction, ignore_index);
    } else if (kind == c10::aten::l1_loss) {
      std::int64_t reduction =
          *HandleConstant<std::int64_t>(node->inputs()[2]->node());

      // Convert to popart reduce values.
      reduction = convertReduceToPopart(reduction);

      // Popart calculates the L1 loss as being the difference from an input to
      // 0. So we have to manually subract the losses first.
      torch::jit::Node *subtract =
          Create_sub(graph, {node->inputs()[0], node->inputs()[1]});
      subtract->insertBefore(node);

      const float scale = 1.0f;
      newNode = Create_l1loss(graph, {subtract->output()}, scale, reduction);

    } else if (kind == c10::aten::mse_loss) {
      std::int64_t reduction =
          *HandleConstant<std::int64_t>(node->inputs()[2]->node());

      // Convert to popart reduce values.
      reduction = convertReduceToPopart(reduction);

      // Subtract X - Y
      torch::jit::Node *subtract =
          Create_sub(graph, {node->inputs()[0], node->inputs()[1]});
      subtract->insertBefore(node);

      // Square it.
      torch::jit::Node *square =
          Create_mul(graph, {subtract->output(), subtract->output()});
      square->insertAfter(subtract);

      torch::jit::Node *finalNode = square;

      if (reduction == 0) {
        // Sum
        finalNode = Create_sum(graph, {square->output()});
        finalNode->insertAfter(square);
      } else if (reduction == 1) {
        // Mean
        finalNode = Create_mean(graph, {square->output()});
        finalNode->insertAfter(square);
      }

      newNode = Create_identityloss(graph, {finalNode->output()}, reduction);

    } else if (kind == Symbols::poptorch::begin_ipu_block) {
      // This could maybe be improved. Can we add attributes on the frontend?
      // TODO(tbd)
      newNode = graph.create(
          c10::Symbol::fromQualString("poptorch::begin_ipu_block"));

      // Convert the prim::Constant into an attribute.
      std::int64_t ipu_id =
          *HandleConstant<std::int64_t>(node->input()->node());
      newNode->i_(c10::Symbol::fromQualString("attr::ipu"), ipu_id);
    } else if (kind == c10::aten::mul) {
      torch::jit::Value *other = node->inputs()[1];

      std::optional<float> asScalar = HandleConstant<float>(other->node());

      if (asScalar) {
        torch::jit::Node *asConstant =
            Create_ConstantFloat(graph, {*asScalar}, {1});
        asConstant->insertBefore(node);

        other->replaceAllUsesWith(asConstant->output());

        // Mark it for deletion.
        toDelete.insert(other->node());

        // Use the popart constant instead.
        other = asConstant->output();
      }

      newNode = Create_mul(graph, {node->inputs()[0], other});
    } else if (kind == c10::aten::select) {
      // clang-format off
      // aten::select(Tensor self, int dim, int index) -> Tensor

      // Note: there is also this overload which is not supported at the moment
      // aten::select(Tensor[] list, int idx) -> Tensor
      // clang-format on

      std::int64_t dim =
          *HandleConstant<std::int64_t>(node->inputs()[1]->node());

      std::int64_t index =
          *HandleConstant<std::int64_t>(node->inputs()[2]->node());

      newNode =
          Create_slice(graph, {node->inputs()[0]}, {index + 1}, {index}, {dim});
    } else if (kind == c10::aten::slice) {
      // clang-format off
      // aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor // NOLINT
      // clang-format on

      std::int64_t dim =
          *HandleConstant<std::int64_t>(node->inputs()[1]->node());

      std::int64_t start =
          *HandleConstant<std::int64_t>(node->inputs()[2]->node());

      std::int64_t end =
          *HandleConstant<std::int64_t>(node->inputs()[3]->node());
      if (end == 9223372036854775807 || end == -1) {
        c10::TensorTypePtr asTensor =
            node->inputs()[0]->type()->cast<c10::TensorType>();
        c10::VaryingShape dims = asTensor->sizes();

        end = *dims[dim];
      }

      newNode = Create_slice(graph, {node->inputs()[0]}, {end}, {start}, {dim});
    } else if (kind == c10::aten::permute) {
      // clang-format off
      // aten::permute(Tensor self, int[] dims) -> Tensor
      // clang-format on

      std::vector<std::int64_t> permutation =
          HandleList<std::int64_t>(node->inputs()[1]->node());

      c10::TensorTypePtr asTensor =
          node->inputs()[0]->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();

      std::for_each(permutation.begin(), permutation.end(),
                    [&](std::int64_t &val) {
                      if (val < 0) {
                        val = *dims.size() + val;
                      }
                    });

      newNode = Create_transpose(graph, {node->inputs()[0]}, permutation);
    } else if (kind == c10::aten::contiguous) {
      // clang-format off
      // aten::contiguous(Tensor self, *, MemoryFormat memory_format=contiguous_format) -> Tensor // NOLINT
      // Returns a copy of the tensor but in contiguous memory.
      // clang-format on

      node->output()->replaceAllUsesWith(node->inputs()[0]);
      toDelete.insert(node);
    } else if (kind == c10::aten::transpose) {
      // clang-format off
      // aten::transpose(Tensor self, int dim0, int dim1) -> Tensor
      // clang-format on
      std::int64_t dim0 =
          *HandleConstant<std::int64_t>(node->inputs()[1]->node());

      std::int64_t dim1 =
          *HandleConstant<std::int64_t>(node->inputs()[2]->node());

      c10::TensorTypePtr asTensor =
          node->inputs()[0]->type()->cast<c10::TensorType>();
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

      newNode = Create_transpose(graph, {node->inputs()[0]}, permutation);
    } else if (kind == c10::aten::div) {
      torch::jit::Value *other = node->inputs()[1];
      std::optional<float> asScalar = HandleConstant<float>(other->node());

      if (asScalar) {
        torch::jit::Node *asConstant =
            Create_ConstantFloat(graph, {*asScalar}, {1});
        asConstant->insertBefore(node);

        other->replaceAllUsesWith(asConstant->output());

        // Mark it for deletion.
        toDelete.insert(other->node());

        // Use the popart constant instead.
        other = asConstant->output();
      }

      newNode = Create_div(graph, {node->inputs()[0], node->inputs()[1]});
    } else if (kind == c10::aten::embedding) {
      // aten::embedding(Tensor weight, Tensor indices, int padding_idx, bool
      // scale_grad_by_freq, bool sparse) -> Tensor

      bool scale_grad_by_freq =
          *HandleConstant<bool>(node->inputs()[3]->node());
      bool sparse = *HandleConstant<bool>(node->inputs()[4]->node());

      if (scale_grad_by_freq || sparse) {
        std::cout << "Unsupported aten::embedding operation" << std::endl;
        newNode->dump();
        exit(0);
      }

      newNode = Create_gather(graph, {node->inputs()[0], node->inputs()[1]}, 0);
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
          node->inputs()[0]->type()->expect<c10::TensorType>();

      c10::ScalarType outAsScalar = *outputType->scalarType();
      c10::ScalarType inAsScalar = *inputType->scalarType();

      // Remove the node if casting to the same type.
      if (outAsScalar == inAsScalar) {
        node->output()->replaceAllUsesWith(node->inputs()[0]);
        toDelete.insert(node);
        continue;
      }

      // Otherwise cast as normal.
      newNode = Create_Cast(graph, node->inputs()[0], outAsScalar);

    } else if (kind == c10::aten::rsub) {
      // clang-format off
      // Tensor aten::rsub(const Tensor& self, const Tensor& other, Scalar alpha) // NOLINT
      // clang-format on
      // We are ignoring alpha here.

      torch::jit::Value *other = node->inputs()[1];

      std::optional<float> asScalar = HandleConstant<float>(other->node());

      // This operation can also take a scalar for other. If it is that overload
      // then we have to add it as a popart scalar and work with that instead.
      if (asScalar) {
        torch::jit::Node *asConstant =
            Create_ConstantFloat(graph, {*asScalar}, {1});
        asConstant->insertBefore(node);

        other->replaceAllUsesWith(asConstant->output());

        // Mark it for deletion.
        toDelete.insert(other->node());

        // Use the popart constant instead.
        other = asConstant->output();
      }

      newNode = Create_sub(graph, {other, node->inputs()[0]});
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
      std::size_t end =
          *HandleConstant<std::int64_t>(node->inputs()[0]->node());
      for (std::size_t start = 0; start < end; ++start) {
        vals.push_back(start);
      }

      newNode = Create_ConstantInt(graph, vals,
                                   {static_cast<std::int64_t>(vals.size())});
    } else if (kind == Symbols::poptorch::identity_loss) {
      std::int64_t reduction =
          *HandleConstant<std::int64_t>(node->inputs()[1]->node());

      newNode = Create_identityloss(graph, {node->inputs()[0]}, reduction);
    } else if (kind == c10::aten::layer_norm) {
      // clang-format off
      // aten::layer_norm(Tensor input,int[] normalized_shape, Tensor? weight,
      //                Tensor? bias, float eps, bool cudnn_enable) -> Tensor
      // clang-format on

      // Tensor to normalise.
      torch::jit::Value *X = node->inputs()[0];

      // Bias to add
      torch::jit::Value *gamma = node->inputs()[2];

      // Weight to multiply.
      torch::jit::Value *beta = node->inputs()[3];

      const float epsilon = *HandleConstant<float>(node->inputs()[4]->node());

      // Pytorch normalizes across arbitrary number of dimensions from the end.
      // We flatten into a [M, N] array and normalize the N.
      std::vector<std::int64_t> normalizedShape =
          HandleList<int64_t>(node->inputs()[1]->node());
      const std::int64_t axis = -normalizedShape.size();

      // Flatten into [M, N]
      torch::jit::Node *flatten = Create_flatten(graph, {X}, axis);
      flatten->insertBefore(node);

      // Normalize.
      torch::jit::Node *normalize = Create_groupnormalization(
          graph, {flatten->output(), gamma, beta}, 1, epsilon);
      normalize->insertAfter(flatten);

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
          *HandleConstant<std::int64_t>(node->inputs()[1]->node());

      newNode = Create_identityloss(graph, {node->inputs()[0]}, reduction);
    } else if (kind == c10::aten::split || kind == c10::aten::chunk) {
      // clang-format off
      // aten::split(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]"
      // aten::split(Tensor self, int split_sizes, int dim=0) -> Tensor[]"
      // aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]
      // clang-format on

      // Get the shape of the input.
      c10::TensorTypePtr asTensor =
          node->inputs()[0]->type()->expect<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();

      // Pythonic axis translation.
      const std::int64_t dim =
          *HandleConstant<std::int64_t>(node->inputs()[2]->node());
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
        newNode = Create_slice(graph, {node->inputs()[0]}, {index + sliceSize},
                               {index}, {axis});

        // Add the slice to the graph.
        newNode->insertBefore(node);
        slices.push_back(newNode->output());

        // Move along in the vector dimension.
        index += sliceSize;
      }

      newNode = graph.create(at::prim::ListConstruct, slices);

    } else if (kind == c10::aten::eq) {
      // clang-format off
      // "aten::eq(Tensor self, Tensor other) -> Tensor"
      // "aten::eq(Tensor self, Scalar other) -> Tensor"
      // clang-format on

      torch::jit::Value *other = node->inputs()[1];

      std::optional<std::int64_t> isInt =
          HandleConstant<std::int64_t>(other->node());

      if (isInt) {
        // Add int as tensor.
        torch::jit::Node *c = Create_ConstantInt(graph, {*isInt}, {1});
        c->insertBefore(node);
        other = c->output();
      }

      newNode = Create_equal(graph, {node->inputs()[0], other});

    } else if (kind == c10::aten::masked_fill) {
      // clang-format off
      // Derived from documentation
      // aten::masked_fill(Tensor self, Tensor mask, Tensor other) -> Tensor
      // clang-format on

      // Apply by performing the following operation
      // inverseMask = -(mask - 1)
      // self * inverseMask + mask * other

      // Cast the mask to int32.
      torch::jit::Node *mask = Create_Cast(graph, node->inputs()[1], c10::kInt);
      mask->insertBefore(node);

      // Create an inverse mask via -(mask - 1)
      torch::jit::Node *negativeOne = Create_ConstantInt(graph, {-1}, {1});
      negativeOne->insertBefore(node);

      torch::jit::Node *inverseMask =
          Create_add(graph, {mask->output(), negativeOne->output()});
      inverseMask->insertBefore(node);

      inverseMask = Create_neg(graph, {inverseMask->output()});
      inverseMask->insertBefore(node);

      // Prepare input and update
      mask = Create_Cast(graph, node->inputs()[1], c10::kFloat);
      mask->insertBefore(node);

      float otherAsConst = *HandleConstant<float>(node->inputs()[2]->node());
      torch::jit::Node *other =
          Create_ConstantFloat(graph, {otherAsConst}, {1});
      other->insertBefore(node);

      torch::jit::Node *update =
          Create_mul(graph, {mask->output(), other->output()});
      update->insertBefore(node);

      // Create holes in the original so we can add into it.
      inverseMask = Create_Cast(graph, inverseMask->output(), c10::kFloat);
      inverseMask->insertBefore(node);

      torch::jit::Node *self =
          Create_mul(graph, {node->inputs()[0], inverseMask->output()});
      self->insertBefore(node);

      newNode = Create_add(graph, {self->output(), update->output()});
    } else if (kind == c10::aten::rsqrt) {
      // rsqrt =  1 / sqrt(x)
      torch::jit::Node *sqrt = Create_sqrt(graph, {node->input()});
      sqrt->insertBefore(node);

      newNode = Create_reciprocal(graph, {sqrt->output()});
    } else if (kind == c10::aten::expm1) {
      // expm1 = exp(x) - 1

      // exp(x)
      torch::jit::Node *exp = Create_exp(graph, {node->input()});
      exp->insertBefore(node);

      // Add the one constant
      torch::jit::Node *one = Create_ConstantFloat(graph, {1.0}, {});
      one->insertBefore(node);

      newNode = Create_sub(graph, {exp->output(), one->output()});
    } else if (kind == c10::aten::trunc) {
      // Drop the exponent by casting to int and back.
      torch::jit::Node *toInt = Create_Cast(graph, node->input(), c10::kInt);
      toInt->insertBefore(node);

      newNode = Create_Cast(graph, toInt->output(), c10::kFloat);

    } else if (kind == c10::aten::frac) {
      // Frac(x) = x - trunc(x)

      // Drop the exponent by casting to int and back.
      torch::jit::Node *toInt = Create_Cast(graph, node->input(), c10::kInt);
      toInt->insertBefore(node);

      torch::jit::Node *trunc =
          Create_Cast(graph, toInt->output(), c10::kFloat);
      trunc->insertBefore(node);

      newNode = Create_sub(graph, {node->input(), trunc->output()});
    } else if (kind == c10::aten::round) {
      // round(x) = trunc(x + sign(x)*0.5)

      // Add 0.5 as constant.
      torch::jit::Node *zeroPointFive = Create_ConstantFloat(graph, {0.5}, {});
      zeroPointFive->insertBefore(node);

      torch::jit::Node *sign = Create_sign(graph, {node->input()});
      sign->insertBefore(node);

      torch::jit::Node *broadcastBySign =
          Create_mul(graph, {sign->output(), zeroPointFive->output()});
      broadcastBySign->insertBefore(node);

      torch::jit::Node *addition =
          Create_add(graph, {node->input(), broadcastBySign->output()});
      addition->insertBefore(node);

      // Drop the exponent by casting to int and back.
      torch::jit::Node *toInt =
          Create_Cast(graph, addition->output(), c10::kInt);
      toInt->insertBefore(node);

      newNode = Create_Cast(graph, toInt->output(), c10::kFloat);
    } else if (kind == c10::aten::floor_divide) {
      // aten::floor_divide(Tensor x, Tensor y) -> Tensor
      // floor_divide(x, y) = floor(x)/floor(y)

      torch::jit::Node *x = Create_floor(graph, {node->inputs()[0]});
      torch::jit::Node *y = Create_floor(graph, {node->inputs()[1]});
      x->insertBefore(node);
      y->insertBefore(node);

      newNode = Create_div(graph, {x->output(), y->output()});

    } else if (kind == c10::aten::true_divide) {
      // aten::true_divide(Tensor x, Tensor y) -> Tensor
      // true_divide(x, y) = (float)x / (float)y

      torch::jit::Node *x = Create_Cast(graph, node->inputs()[0], c10::kFloat);
      x->insertBefore(node);

      torch::jit::Node *y = Create_Cast(graph, node->inputs()[1], c10::kFloat);
      y->insertBefore(node);

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
        flatten->insertBefore(node);
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
        flatten->insertBefore(node);
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
    }

    // If we have a new node add it and replace the old use.
    if (newNode) {
      newNode->insertBefore(node);

      // Mark this node for deletion.
      toDelete.insert(node);

      if (node->hasUses()) {
        torch::jit::Value *newVal = newNode->output();
        torch::jit::Value *oldVal = node->output();

        // Take the type of the old value.
        newVal->setType(oldVal->type());

        // Replace the old value with the new one.
        oldVal->replaceAllUsesWith(newVal);
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
