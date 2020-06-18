// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poptorch/PopartCanonicalization.hpp>

#include <optional>
#include <string>
#include <torch/csrc/jit/ir/ir.h>
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
        kind == c10::aten::reshape) {
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
    }
#define HANDLE(Index, Type) *HandleConstant<Type>(node->inputs()[Index]->node())
#define PARAM(Index) node->inputs()[Index]
#define COMMA ,
#define NONE

// Handle all supported scalar values and pass the correct C++ type to the given
// body.
#define ANY_SCALAR_CONSTANT_HANDLER(Value, Body)                               \
  c10::TensorTypePtr asTensor = Value->type()->cast<c10::TensorType>();        \
  at::ScalarType type = *asTensor->scalarType();                               \
  if (type == at::ScalarType::Float || type == at::ScalarType::Double) {       \
    Body(float)                                                                \
  } else if (type == at::ScalarType::Int || type == at::ScalarType::Long ||    \
             type == at::ScalarType::Bool) {                                   \
    Body(int)                                                                  \
  }                                                                            \
// Many binary element wise operations contained a fused "Alpha" component. The
// form of this is A (+, -) B * alpha. Most of the time this will be zero so can
// be skipped but it could be non-zero and must be handled.

// We have a helper macro to insert the alpha value if needed.
#define ALPHA_BODY(Type)                                                       \
  Type alphaAsScalar = *HandleConstant<std::int64_t>(alphaValue->node());      \
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
  ANY_SCALAR_CONSTANT_HANDLER(valueToMultiply, ALPHA_BODY)                     \
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

      // TODO: These will have to be checked if they are actual tensors in the
      // future.
      std::vector<torch::jit::Value *> inputTensors{input, weight, bias,
                                                    running_mean, running_var};

      float momentum = *HandleConstant<float>(node->inputs()[6]->node());
      float epsilon = *HandleConstant<float>(node->inputs()[7]->node());

      newNode = poptorch::Create_batchnormalization(graph, inputTensors, 1,
                                                    epsilon, momentum);

    } else if (kind == c10::aten::max_pool2d) {
      // clang-format off
      /*
        aten::max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[]
        padding, int[] dilation, bool ceil_mode) -> Tensor
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

      newNode = poptorch::Create_maxpool(graph, {node->inputs()[0]}, 1,
                                         kernel_size, padding, 0, stride);
    } else if (kind == c10::aten::adaptive_avg_pool2d) {
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
      // TODO: Handle type.
      // clang-format on
      c10::TensorTypePtr asTensor =
          node->outputs()[0]->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();
      std::vector<std::int64_t> operationShape;

      for (auto optionalInt : *dims.sizes()) {
        operationShape.push_back(*optionalInt);
      }

      newNode = Create_ConstantFloat(graph, {1.0f}, operationShape);
    } else if (kind == c10::aten::zeros) {
      // clang-format off
      // aten::zeros(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor // NOLINT
      // TODO: Handle type.
      // clang-format on
      c10::TensorTypePtr asTensor =
          node->outputs()[0]->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = asTensor->sizes();
      std::vector<std::int64_t> operationShape;

      for (auto optionalInt : *dims.sizes()) {
        operationShape.push_back(*optionalInt);
      }

      newNode = Create_ConstantInt(graph, {0}, operationShape);
    } else if (kind == c10::aten::to) {
      // clang-format off
      // aten::to(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)" // NOLINT
      // aten::to(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)" // NOLINT
      // aten::to(Tensor(a) self, bool non_blocking=False, bool copy=False) -> Tensor(a|b)" // NOLINT
      // clang-format on

      // In BERT the cast is to the same type so ignore for now.
      node->output()->replaceAllUsesWith(node->inputs()[0]);
      toDelete.insert(node);
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
