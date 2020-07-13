// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "torch/csrc/jit/ir/ir.h"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/Utils.hpp"

#include "PoptorchSymbols.hpp"
#include "popart_canonicalization/PopartCanonicalizationUtils.hpp"

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

class CanonicalizeImpl {
public:
  void run(torch::jit::Graph *graph);

private:
  // When we transform a node mark it for deletion, this will also clean up
  // unused users afterwards.
  std::unordered_set<torch::jit::Node *> _to_delete;

  // This handles the case of both `prim::ListConstruct`
  // and 'prim::Constant[value=[x, y, z]]'.
  template <typename T> std::vector<T> handleList(torch::jit::Node *node);

  template <typename T>
  std::vector<T> handleListConstruct(torch::jit::Node *node);

  static std::vector<torch::jit::Value *>
  handleTensorList(torch::jit::Node *node);

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
      node->input(0)->type()->cast<c10::TensorType>();
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

std::vector<torch::jit::Value *>
CanonicalizeImpl::handleTensorList(torch::jit::Node *node) {
  std::vector<torch::jit::Value *> result;

  // Just convert the node->inputs array ref to vector and return it.
  for (torch::jit::Value *value : node->inputs()) {
    result.push_back(value);
  }

  return result;
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
  reduceHelperDimensionCreator(node->inputs()[Index])

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

#include "CanonicalizationOps.h.inc"

#undef OP_CONVERTOR
#undef PARAM
#undef COMMA
#undef HANDLE
#undef NONE
#undef ALPHA

    // NOLINTNEXTLINE
    else if (kind == c10::aten::batch_norm) {
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
      std::vector<std::int64_t> original_shape =
          shapeFromTensor(node->input(0));

      // New 4D shape to perform the operation with.
      std::vector<std::int64_t> new_shape = original_shape;

      // Turn the shape into a 4D tensor.
      if (original_shape.size() == 2) {
        // Add two singletons to pad to 4D.
        new_shape.push_back(1);
        new_shape.push_back(1);
      } else if (original_shape.size() == 3) {
        // Add one singleton to get to 4D.
        new_shape.push_back(1);
      } else if (original_shape.size() == 5) {
        // Flatten last two dimensions to reduce to 4.
        new_shape[3] *= new_shape[4];
        new_shape.pop_back();
      }

      // Input is value at 0th position.
      torch::jit::Value *input = node->input(0);

      // Reshape to 4D if needed.
      if (original_shape.size() != 4) {
        torch::jit::Node *reshape_in = createReshape(graph, input, new_shape);
        input = reshape_in->output();
      }

      torch::jit::Value *weight = node->input(1);
      torch::jit::Value *bias = node->input(2);
      torch::jit::Value *running_mean = node->input(3);
      torch::jit::Value *running_var = node->input(4);

      // TODO(T22645): These will have to be checked if they are actual tensors
      // in the future.
      std::vector<torch::jit::Value *> input_tensors{input, weight, bias,
                                                     running_mean, running_var};

      float momentum = *handleConstant<float>(node->input(6)->node());
      float epsilon = *handleConstant<float>(node->input(7)->node());

      new_node = poptorch::createBatchnormalization(graph, input_tensors, 1,
                                                    epsilon, momentum);

      // If we reshaped, reshape back.
      if (original_shape.size() != 4) {
        // Add the batch norm.

        // This is now the new node.
        new_node = createReshape(graph, new_node->output(), original_shape);
      }

    } else if (kind == symbols::poptorch::begin_ipu_block) {
      // This could maybe be improved. Can we add attributes on the frontend?
      // TODO(tbd)
      new_node = createAndInsertNode(
          graph, c10::Symbol::fromQualString("poptorch::begin_ipu_block"), {},
          node->outputs().size());

      // Convert the prim::Constant into an attribute.
      std::int64_t ipu_id =
          *handleConstant<std::int64_t>(node->input()->node());
      new_node->i_(c10::Symbol::fromQualString("attr::ipu"), ipu_id);
    } else if (kind == c10::aten::select) {
      // clang-format off
      // aten::select(Tensor self, int dim, int index) -> Tensor

      // Note: there is also this overload which is not supported at the moment
      // aten::select(Tensor[] list, int idx) -> Tensor
      // clang-format on

      std::int64_t dim = *handleConstant<std::int64_t>(node->input(1)->node());

      std::int64_t index =
          *handleConstant<std::int64_t>(node->input(2)->node());

      new_node =
          createSlice(graph, {node->input(0)}, {index + 1}, {index}, {dim});
    } else if (kind == c10::aten::slice) {
      // clang-format off
      // aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor // NOLINT
      // clang-format on

      std::int64_t dim = *handleConstant<std::int64_t>(node->input(1)->node());

      std::int64_t start =
          *handleConstant<std::int64_t>(node->input(2)->node());

      std::int64_t end = *handleConstant<std::int64_t>(node->input(3)->node());
      if (end == 9223372036854775807 || end == -1) {
        c10::TensorTypePtr as_tensor =
            node->input(0)->type()->cast<c10::TensorType>();
        c10::VaryingShape dims = as_tensor->sizes();

        end = *dims[dim];
      }

      new_node = createSlice(graph, {node->input(0)}, {end}, {start}, {dim});
    } else if (kind == c10::aten::permute) {
      // clang-format off
      // aten::permute(Tensor self, int[] dims) -> Tensor
      // clang-format on

      std::vector<std::int64_t> permutation =
          handleList<std::int64_t>(node->input(1)->node());

      c10::TensorTypePtr as_tensor =
          node->input(0)->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = as_tensor->sizes();

      std::for_each(permutation.begin(), permutation.end(),
                    [&](std::int64_t &val) {
                      if (val < 0) {
                        val = *dims.size() + val;
                      }
                    });

      new_node = createTranspose(graph, {node->input(0)}, permutation);
    } else if (kind == c10::aten::contiguous || kind == c10::aten::detach) {
      // clang-format off
      // aten::contiguous(Tensor self, *, MemoryFormat memory_format=contiguous_format) -> Tensor // NOLINT
      // Returns a copy of the tensor but in contiguous memory.
      //
      // aten::detach(Tensor self) -> Tensor
      // Returns the tensor
      // clang-format on

      node->output()->replaceAllUsesWith(node->input(0));
      _to_delete.insert(node);
    } else if (kind == c10::aten::transpose) {
      // clang-format off
      // aten::transpose(Tensor self, int dim0, int dim1) -> Tensor
      // clang-format on
      std::int64_t dim0 = *handleConstant<std::int64_t>(node->input(1)->node());

      std::int64_t dim1 = *handleConstant<std::int64_t>(node->input(2)->node());

      c10::TensorTypePtr as_tensor =
          node->input(0)->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = as_tensor->sizes();

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

      new_node = createTranspose(graph, {node->input(0)}, permutation);
    } else if (kind == c10::aten::embedding) {
      // aten::embedding(Tensor weight, Tensor indices, int padding_idx, bool
      // scale_grad_by_freq, bool sparse) -> Tensor

      bool scale_grad_by_freq = *handleConstant<bool>(node->input(3)->node());
      bool sparse = *handleConstant<bool>(node->input(4)->node());

      if (scale_grad_by_freq || sparse) {
        std::cout << "Unsupported aten::embedding operation" << std::endl;
        node->dump();
        exit(0);
      }

      new_node = createGather(graph, {node->input(0), node->input(1)}, 0);
    } else if (kind == c10::aten::ones) {
      // clang-format off
      // aten::ones(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor // NOLINT
      // clang-format on
      c10::TensorTypePtr as_tensor =
          node->outputs()[0]->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = as_tensor->sizes();
      std::vector<std::int64_t> operation_shape;

      for (auto optional_int : *dims.sizes()) {
        operation_shape.push_back(*optional_int);
      }

      switch (*as_tensor->scalarType()) {
      case c10::ScalarType::Int:
      case c10::ScalarType::Long: {
        new_node = createConstantInt(graph, {1}, operation_shape);
        break;
      }
      case c10::ScalarType::Float: {
        new_node = createConstantFloat(graph, {1.0}, operation_shape);
        break;
      }
      default:
        ERROR("aten::ones of type " << c10::toString(*as_tensor->scalarType())
                                    << " not supported");
      }
    } else if (kind == c10::aten::zeros) {
      // clang-format off
      // aten::zeros(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor // NOLINT
      // clang-format on
      c10::TensorTypePtr as_tensor =
          node->outputs()[0]->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = as_tensor->sizes();
      std::vector<std::int64_t> operation_shape;

      for (auto optional_int : *dims.sizes()) {
        operation_shape.push_back(*optional_int);
      }

      switch (*as_tensor->scalarType()) {
      case c10::ScalarType::Int:
      case c10::ScalarType::Long: {
        new_node = createConstantInt(graph, {0}, operation_shape);
        break;
      }
      case c10::ScalarType::Float: {
        new_node = createConstantFloat(graph, {0.0}, operation_shape);
        break;
      }
      default:
        ERROR("aten::zeros of type " << c10::toString(*as_tensor->scalarType())
                                     << " not supported");
      }
    } else if (kind == c10::aten::to) {
      auto tensor_type = node->input(0)->type()->cast<c10::TensorType>();
      ERROR_ON_MSG(
          !tensor_type,
          "Casting from a non-tensor type not supported, in an aten::to.");

      // clang-format off
      // aten::to(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)" // NOLINT
      // aten::to(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)" // NOLINT
      // aten::to(Tensor(a) self, [args without dtype])
      // clang-format on

      std::optional<c10::ScalarType> cast_to;
      if (node->input(1)->type()->cast<c10::DeviceObjType>() ||
          node->input(1)->type()->cast<c10::IntType>()) {
        auto output_type = node->output(0)->type()->expect<c10::TensorType>();
        cast_to = *output_type->scalarType();
      }

      if (cast_to.has_value()) {
        // Avoid promoting to an unsupported type
        if (*cast_to == at::ScalarType::Double) {
          cast_to = at::ScalarType::Float;
        } else if (*cast_to == at::ScalarType::Long) {
          cast_to = at::ScalarType::Int;
        }
      }

      if (!cast_to.has_value() || cast_to == *tensor_type->scalarType()) {
        // NOOP
        logging::trace("Ignoring type cast to same type, {}, {}", *cast_to,
                       *tensor_type->scalarType());
        node->output()->replaceAllUsesWith(node->input(0));
        _to_delete.insert(node);
      } else {
        new_node = createCast(graph, node->input(0), *cast_to);
      }
    } else if (kind == c10::aten::rsub) {
      // clang-format off
      // Tensor aten::rsub(const Tensor& self, const Tensor& other, Scalar alpha) // NOLINT
      // clang-format on
      // We are ignoring alpha here.

      torch::jit::Value *other = node->input(1);

      std::optional<float> as_scalar = handleConstant<float>(other->node());

      // This operation can also take a scalar for other. If it is that overload
      // then we have to add it as a popart scalar and work with that instead.
      if (as_scalar) {
        torch::jit::Node *as_constant =
            createConstantFloat(graph, {*as_scalar}, {1});

        other->replaceAllUsesWith(as_constant->output());

        // Mark it for deletion.
        _to_delete.insert(other->node());

        // Use the popart constant instead.
        other = as_constant->output();
      }

      new_node = createSub(graph, {other, node->input(0)});
    } else if (kind == c10::aten::arange) {
      // clang-format off
      // aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory) // NOLINT
      // aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory) // NOLINT
      // clang-format on

      if (node->inputs().size() != 5) {
        std::cerr << "Unsupported arrange op";
        node->dump();
      }

      std::vector<std::int64_t> vals;
      std::size_t end = *handleConstant<std::int64_t>(node->input(0)->node());
      for (std::size_t start = 0; start < end; ++start) {
        vals.push_back(start);
      }

      new_node = createConstantInt(graph, vals,
                                   {static_cast<std::int64_t>(vals.size())});
    } else if (kind == symbols::poptorch::identity_loss) {
      std::int64_t reduction =
          *handleConstant<std::int64_t>(node->input(1)->node());

      new_node = createIdentityloss(graph, {node->input(0)}, reduction);
    } else if (kind == c10::aten::layer_norm) {
      // clang-format off
      // aten::layer_norm(Tensor input,int[] normalized_shape, Tensor? weight,
      //                Tensor? bias, float eps, bool cudnn_enable) -> Tensor
      // clang-format on

      // Tensor to normalise.
      torch::jit::Value *x = node->input(0);

      // Bias to add
      torch::jit::Value *gamma = node->input(2);

      // Weight to multiply.
      torch::jit::Value *beta = node->input(3);

      const float epsilon = *handleConstant<float>(node->input(4)->node());

      // Pytorch normalizes across arbitrary number of dimensions from the end.
      // We flatten into a [M, N] array and normalize the N.

      std::vector<std::int64_t> normalized_shape =
          handleList<int64_t>(node->input(1)->node());

      // Opset 9 Flatten does not handle negative axis
      auto tensor_type = x->type()->expect<c10::TensorType>();
      const std::int64_t axis = *tensor_type->dim() - normalized_shape.size();

      // Flatten into [M, N]
      torch::jit::Node *flatten = createFlatten(graph, {x}, axis);

      // Normalize.
      torch::jit::Node *normalize = createGroupnormalization(
          graph, {flatten->output(), gamma, beta}, 1, epsilon);

      // Reshape back into the expected shape.
      c10::TensorTypePtr converted_to_tensor =
          node->output()->type()->cast<c10::TensorType>();
      c10::VaryingShape dims = converted_to_tensor->sizes();

      std::vector<std::int64_t> original_shape;

      for (auto optional_int : *dims.sizes()) {
        original_shape.push_back(*optional_int);
      }

      // Perform the reshape.
      new_node = createReshape(graph, normalize->output(), original_shape);

    } else if (kind == symbols::poptorch::identity_loss) {
      std::int64_t reduction =
          *handleConstant<std::int64_t>(node->input(1)->node());

      new_node = createIdentityloss(graph, {node->input(0)}, reduction);
    } else if (kind == c10::aten::split || kind == c10::aten::chunk) {
      // clang-format off
      // aten::split(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]"
      // aten::split(Tensor self, int split_sizes, int dim=0) -> Tensor[]"
      // aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]
      // clang-format on

      // Get the shape of the input.
      c10::TensorTypePtr as_tensor =
          node->input(0)->type()->expect<c10::TensorType>();
      c10::VaryingShape dims = as_tensor->sizes();

      // Pythonic axis translation.
      const std::int64_t dim =
          *handleConstant<std::int64_t>(node->input(2)->node());
      const std::int64_t axis = dim >= 0 ? dim : *dims.size() + dim;

      // Size of each split ignoring the remainder at the end.
      std::vector<std::int64_t> size_of_each_split;

      // Split size can either be the number of splits or the size of the
      // splits.
      std::optional<std::int64_t> split_size =
          handleConstant<std::int64_t>(node->input(1)->node());

      if (kind == c10::aten::chunk) {
        // Chunk takes in the *number of chunks*. Canonicalise it to *size of
        // chunks*.
        ERROR_ON_MSG(
            !split_size,
            "Aten chunk node does not have a integer number of chunks!");
        std::int64_t slice_size = *dims[axis] / *split_size;
        for (int i = 0; i < *split_size; ++i) {
          size_of_each_split.push_back(slice_size);
        }

        // Add an extra slice for the remainder.
        if (*dims[axis] % *split_size != 0) {
          size_of_each_split.push_back(*dims[axis] % *split_size);
        }
      } else if (split_size) {
        // Split takes in the size of each chunk.
        std::int64_t slice_size = *split_size;
        for (int i = 0; i < *dims[axis] / slice_size; ++i) {
          size_of_each_split.push_back(slice_size);
        }

        // Add an extra slice for the remainder.
        if (*dims[axis] % *split_size != 0) {
          size_of_each_split.push_back(*dims[axis] % *split_size);
        }
      } else {
        size_of_each_split = handleList<std::int64_t>(node->input(1)->node());
      }

      // Rolling index to track where we are in the tensor.
      std::int64_t index = 0;

      // The result of each slice.
      std::vector<torch::jit::Value *> slices;

      // Slice up according to the canonicalised split vector.
      for (std::int64_t slice_size : size_of_each_split) {
        // Create a slice.
        new_node = createSlice(graph, {node->input(0)}, {index + slice_size},
                               {index}, {axis});

        // Add the slice to the graph.
        slices.push_back(new_node->output());

        // Move along in the vector dimension.
        index += slice_size;
      }

      new_node = createAndInsertNode(graph, at::prim::ListConstruct, slices);
    } else if (kind == c10::aten::masked_fill) {
      // clang-format off
      // Derived from documentation
      // aten::masked_fill(Tensor self, Tensor mask, Tensor other) -> Tensor
      // clang-format on

      // Apply by performing the following operation
      // inverseMask = -(mask - 1)
      // self * inverseMask + mask * other

      // Cast the mask to int32.
      torch::jit::Node *mask = createCast(graph, node->input(1), c10::kInt);

      // Create an inverse mask via -(mask - 1)
      torch::jit::Node *negative_one = createConstantInt(graph, {-1}, {1});

      torch::jit::Node *inverse_mask =
          createAdd(graph, {mask->output(), negative_one->output()});

      inverse_mask = createNeg(graph, {inverse_mask->output()});

      // Prepare input and update
      mask = createCast(graph, node->input(1), c10::kFloat);

      float other_as_const = *handleConstant<float>(node->input(2)->node());
      torch::jit::Node *other =
          createConstantFloat(graph, {other_as_const}, {1});

      torch::jit::Node *update =
          createMul(graph, {mask->output(), other->output()});

      // Create holes in the original so we can add into it.
      inverse_mask = createCast(graph, inverse_mask->output(), c10::kFloat);

      torch::jit::Node *self =
          createMul(graph, {node->input(0), inverse_mask->output()});

      new_node = createAdd(graph, {self->output(), update->output()});
    } else if (kind == c10::aten::rsqrt) {
      // rsqrt =  1 / sqrt(x)
      torch::jit::Node *sqrt = createSqrt(graph, {node->input()});

      new_node = createReciprocal(graph, {sqrt->output()});
    } else if (kind == c10::aten::expm1) {
      // expm1 = exp(x) - 1

      // exp(x)
      torch::jit::Node *exp = createExp(graph, {node->input()});

      // Add the one constant
      torch::jit::Node *one = createConstantFloat(graph, {1.0}, {});

      new_node = createSub(graph, {exp->output(), one->output()});
    } else if (kind == c10::aten::trunc) {
      // Drop the exponent by casting to int and back.
      torch::jit::Node *to_int = createCast(graph, node->input(), c10::kInt);

      new_node = createCast(graph, to_int->output(), c10::kFloat);

    } else if (kind == c10::aten::frac) {
      // Frac(x) = x - trunc(x)

      // Drop the exponent by casting to int and back.
      torch::jit::Node *to_int = createCast(graph, node->input(), c10::kInt);

      torch::jit::Node *trunc =
          createCast(graph, to_int->output(), c10::kFloat);

      new_node = createSub(graph, {node->input(), trunc->output()});
    } else if (kind == c10::aten::round) {
      // round(x) = trunc(x + sign(x)*0.5)

      // Add 0.5 as constant.
      torch::jit::Node *zero_point_five = createConstantFloat(graph, {0.5}, {});

      torch::jit::Node *sign = createSign(graph, {node->input()});

      torch::jit::Node *broadcast_by_sign =
          createMul(graph, {sign->output(), zero_point_five->output()});

      torch::jit::Node *addition =
          createAdd(graph, {node->input(), broadcast_by_sign->output()});

      // Drop the exponent by casting to int and back.
      torch::jit::Node *to_int =
          createCast(graph, addition->output(), c10::kInt);

      new_node = createCast(graph, to_int->output(), c10::kFloat);
    } else if (kind == c10::aten::floor_divide) {
      // aten::floor_divide(Tensor x, Tensor y) -> Tensor
      // floor_divide(x, y) = floor(x)/floor(y)

      torch::jit::Node *x = createFloor(graph, {node->inputs()[0]});
      torch::jit::Node *y = createFloor(graph, {node->inputs()[1]});

      new_node = createDiv(graph, {x->output(), y->output()});

    } else if (kind == c10::aten::true_divide) {
      // aten::true_divide(Tensor x, Tensor y) -> Tensor
      // true_divide(x, y) = (float)x / (float)y

      torch::jit::Node *x = createCast(graph, node->inputs()[0], c10::kFloat);

      torch::jit::Node *y = createCast(graph, node->inputs()[1], c10::kFloat);

      new_node = createDiv(graph, {x->output(), y->output()});
    } else if (kind == c10::aten::argmax || kind == c10::aten::argmin) {
      // clang-format off
      //  aten::argmin(Tensor in, int? dim, int keep_dims) -> Tensor
      //  aten::argmax(Tensor in, int? dim, int keep_dims) -> Tensor
      // dim (int) – the dimension to reduce. If None, the argmax
      //             of the flattened input is returned.
      // clang-format on

      torch::jit::Value *input = node->input(0);
      std::optional<std::int64_t> dim =
          handleConstant<std::int64_t>(node->inputs()[1]->node());
      std::int64_t keep_dim =
          *handleConstant<std::int64_t>(node->inputs()[2]->node());

      // If dim is not provided we will flatten input so just use 0 in that
      // case.
      std::int64_t dim_to_use = 1;

      // Check if dim is NONE.
      if (!dim) {
        torch::jit::Node *flatten =
            createFlatten(graph, {node->inputs()[0]}, 0);
        input = flatten->output();
      } else {
        dim_to_use = *dim;
      }

      // Create the actual argmax/argmin.
      if (kind == c10::aten::argmax) {
        new_node = createArgmax(graph, {input}, dim_to_use, keep_dim);
      } else {
        new_node = createArgmin(graph, {input}, dim_to_use, keep_dim);
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
            createFlatten(graph, {node->inputs()[0]}, 0);
        input = flatten->output();
        axes = {1};
      } else {
        // Case 1.
        // Sometimes the dimensions are just one int.
        std::optional<std::int64_t> as_int =
            handleConstant<std::int64_t>(node->inputs()[1]->node());

        if (as_int) {
          axes.push_back(*as_int);
        } else {
          axes = handleListConstruct<std::int64_t>(node->inputs()[1]->node());
        }

        keepdim = *handleConstant<std::int64_t>(node->inputs()[2]->node());
      }

      // Output the correct reduction.
      if (kind == c10::aten::prod) {
        new_node = createReduceprod(graph, {input}, axes, keepdim);
      } else if (kind == c10::aten::mean) {
        new_node = createReducemean(graph, {input}, axes, keepdim);
      } else if (kind == c10::aten::sum) {
        new_node = createReducesum(graph, {input}, axes, keepdim);
      } else if (kind == c10::aten::logsumexp) {
        new_node = createReducelogsumexp(graph, {input}, axes, keepdim);
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
          *handleConstant<std::int64_t>(node->input(3)->node());

      // Convert to popart reduce values.
      reduction = convertReduceToPopart(reduction);

      // Add the one constant
      torch::jit::Node *one = createConstantFloat(graph, {1.0}, {});

      torch::jit::Node *log_x = createLog(graph, {x});

      // Log(x)*y
      torch::jit::Node *log_x_mul_y = createMul(graph, {y, log_x->output()});

      // Do (1 - y) and (1 - x)
      torch::jit::Node *x_minus_one = createSub(graph, {one->output(), x});
      torch::jit::Node *y_minus_one = createSub(graph, {one->output(), y});

      // Log(1 - x)
      torch::jit::Node *log_x_minus_one =
          createLog(graph, {x_minus_one->output()});

      // (1 -y)*Log(1 - x)
      torch::jit::Node *subs_multiplied =
          createMul(graph, {y_minus_one->output(), log_x_minus_one->output()});

      // Log(x)*y + (1 -y)*Log(1 - x)
      torch::jit::Node *add_terms =
          createAdd(graph, {log_x_mul_y->output(), subs_multiplied->output()});

      torch::jit::Node *final_node = add_terms;

      if (weight->node()->kind() != c10::prim::Constant) {
        final_node = createMul(graph, {add_terms->output(), weight});
      }

      final_node = createNeg(graph, {final_node->output()});
      if (reduction == 0) {
        // Sum
        final_node = createSum(graph, {final_node->output()});
      } else if (reduction == 1) {
        // Mean
        final_node = createMean(graph, {final_node->output()});
      }

      new_node = createIdentityloss(graph, {final_node->output()}, reduction);

    } else if (kind == c10::aten::lstm) {
      // clang-format off
      // aten::lstm(Tensor self, Tensor[] hx, Tensor[] weights, bool bias,
      // int num_layers, float dropout, bool training, bool bidirectional,
      // bool batch_first) -> Tensor, (Tensor, Tensor)
      // clang-format on

      torch::jit::Value *input = node->input(0);

      torch::jit::ArrayRef<torch::jit::Value *> hidden_layers =
          node->input(1)->node()->inputs();
      torch::jit::ArrayRef<torch::jit::Value *> weights_list =
          node->input(2)->node()->inputs();

      bool use_bias = *handleConstant<bool>(node->input(3)->node());
      ERROR_ON_MSG(!use_bias, "LSTM without biases not supported");
      std::int64_t num_layers =
          *handleConstant<std::int64_t>(node->input(4)->node());
      ERROR_ON_MSG(num_layers != 1, "Only LSTM with 1 layer supported");

      float dropout = *handleConstant<float>(node->input(5)->node());
      ERROR_ON_MSG(dropout != 0.0f, "LSTM only supports dropout = 0.0");

      bool bidirectional = *handleConstant<bool>(node->input(7)->node());
      ERROR_ON_MSG(bidirectional, "bidirectional LSTM not supported");

      bool batch_first = *handleConstant<bool>(node->input(8)->node());

      // An LSTM state is made of 4 values
      constexpr std::uint64_t state_size = 4;
      const std::int64_t num_weights =
          *weights_list[0]->type()->cast<c10::TensorType>()->sizes()[0];
      ERROR_ON(num_weights % state_size != 0);
      const std::int64_t num_hidden_layers = num_weights / state_size;

      // def reshape_weights(onnx_weights):
      //    ws = builder.aiOnnx.split([w], 4, 1, [hidden_size] * 4)
      //    ws = [builder.aiOnnx.transpose([i], [0, 2, 1]) for i in ws]
      //    ws = builder.aiOnnx.concat([ws[i] for i in (2, 0, 3, 1)], 0)
      //    return ws
      //
      // Note: onnx weights are in IOFC order while Torch uses IFCO
      //
      // Biases don't need to be transposed
      auto reshape_tensor = [&](torch::jit::Value *values, bool areWeights) {
        const std::uint64_t num_dims_without_batch = areWeights ? 2 : 1;
        std::vector<std::int64_t> shape = shapeFromTensor(values);
        if (shape.size() == num_dims_without_batch) {
          // Add a batch dimension
          shape.insert(shape.begin(), 1);
          torch::jit::Node *reshape = createReshape(graph, values, shape);
          values = reshape->output();
        }
        torch::jit::Node *states =
            createSplit(graph, {values}, state_size, 1,
                        {num_hidden_layers, num_hidden_layers,
                         num_hidden_layers, num_hidden_layers});
        std::vector<torch::jit::Value *> slices;
        for (std::uint64_t i = 0; i < state_size; ++i) {
          if (areWeights) {
            // Weights also need to be transposed
            torch::jit::Node *transposed =
                createTranspose(graph, {states->output(i)}, {0, 2, 1});
            slices.push_back(transposed->output());
          } else {
            slices.push_back(states->output(i));
          }
        }
        torch::jit::Node *concat = createConcat(
            graph, {slices[1], slices[0], slices[2], slices[3]}, 0);
        return concat->output();
      };

      torch::jit::Node *concat_weights =
          createConcat(graph,
                       {reshape_tensor(weights_list[0], true),
                        reshape_tensor(weights_list[1], true)},
                       1);
      torch::jit::Node *combine_biases =
          createAddNotInPlace(graph, reshape_tensor(weights_list[2], false),
                              reshape_tensor(weights_list[3], false));

      torch::jit::Node *concat_states =
          createConcat(graph, {hidden_layers[0], hidden_layers[1]}, 0);

      // Transpose output BSF -> SBF
      if (batch_first) {
        torch::jit::Node *transpose =
            createTranspose(graph, {input}, {1, 0, 2});
        input = transpose->output();
      }
      std::vector<torch::jit::Value *> args;
      args.push_back(input);
      args.push_back(
          concat_weights->output()); // input weights + output_weights
      args.push_back(combine_biases->output()); // biases
      args.push_back(concat_states->output());  // init_states

      torch::jit::Node *lstm = createLstm(graph, args, 1);

      // Keep the last slice from Y
      torch::jit::Node *y_h =
          createSlice(graph, {lstm->output(0)}, {INT_MAX}, {-1}, {0});

      torch::jit::Value *output = lstm->output(0);
      // Transpose output SBF -> BSF
      if (batch_first) {
        torch::jit::Node *transpose =
            createTranspose(graph, {output}, {1, 0, 2});
        output = transpose->output();
      }

      ERROR_ON(node->outputs().size() != 3);
      if (node->hasUses()) {
        replaceOutputUse(node->output(0), output);
        replaceOutputUse(node->output(1), y_h->output());
        replaceOutputUse(node->output(2), lstm->output(1));
      }

      _to_delete.insert(node);
    } else if (kind == c10::aten::ge || kind == c10::aten::le) {
      torch::jit::Node *comparison = nullptr;
      torch::jit::Value *lhs =
          handleParamOrConstantNoCast(graph, node->input(0));
      torch::jit::Value *rhs =
          handleParamOrConstantNoCast(graph, node->input(1));

      // Node will either be < or >.
      if (kind == c10::aten::ge) {
        comparison = createGreater(graph, {lhs, rhs});
      } else {
        comparison = createLess(graph, {lhs, rhs});
      }

      // We do a check for ==
      torch::jit::Node *equal = createEqual(graph, {lhs, rhs});

      // The final node will be a combination of equals and less or greater.
      new_node =
          createLogical_or(graph, {equal->output(), comparison->output()});
    } else if (kind == c10::aten::ne) {
      torch::jit::Value *lhs =
          handleParamOrConstantNoCast(graph, node->input(0));
      torch::jit::Value *rhs =
          handleParamOrConstantNoCast(graph, node->input(1));

      // Not(equal(lhs, rhs))
      torch::jit::Node *equal = createEqual(graph, {lhs, rhs});
      new_node = createLogical_not(graph, {equal->output()});
    }

    // If we have a new node add it and replace the old use.
    if (new_node) {
      // Mark this node for deletion.
      _to_delete.insert(node);
      ERROR_ON(node->outputs().size() != new_node->outputs().size());

      if (node->hasUses()) {
        for (std::uint64_t i = 0; i < node->outputs().size(); ++i) {
          replaceOutputUse(node, new_node, i);
        }
      }
    }
  }

  // Remove any dead nodes.
  for (torch::jit::Node *node : _to_delete) {
    searchAndPossiblyDestroy(node);
  }
}

} // namespace

void canonicalize(torch::jit::Graph *graph) {
  CanonicalizeImpl converter;
  converter.run(graph);
}

} // namespace poptorch
