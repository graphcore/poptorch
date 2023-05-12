// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <ATen/MetaFunctions.h>
#include <algorithm>

#include "poptorch_logging/Logging.hpp"

#include "CommonHelperFunctions.hpp"
#include "TypeInferenceHandler.hpp"
#include <c10/util/intrusive_ptr.h>

namespace poptorch {

constexpr c10::DispatchKeySet meta_keys{c10::DispatchKey::Meta,
                                        c10::DispatchKey::AutogradMeta};

namespace {

c10::Stack copyTensorsFrom(const c10::Stack &meta_stack) {
  c10::Stack tmp_stack;
  tmp_stack.reserve(meta_stack.size());
  std::copy_if(meta_stack.cbegin(), meta_stack.cend(),
               std::back_inserter(tmp_stack), [](const auto &value) {
                 return value.isTensor() || value.isTensorList();
               });
  return tmp_stack;
}

} // namespace

void TypeInferenceHandler::inferOutputTypes(const c10::OperatorHandle &op,
                                            c10::Stack *ipu_stack) {
  const auto schema_key = getSchemaKey(op.schema());
  // Unfortunately, aten::prelu with 1D inputs is broken with the Meta backend:
  // https://github.com/pytorch/pytorch/issues/89560
  // As a workaround, we add a dummy channel dim to the input, and then remove
  // it again afterwards
  const bool is_prelu =
      schema_key == "aten::prelu" || schema_key == "aten::_prelu_kernel";
  // Create a new operand stack with meta tensors
  c10::Stack meta_stack = createMetaStack(*ipu_stack, schema_key, is_prelu);

  // redispatchBoxed drops all function inputs from the stack. Meta stack is
  // the only owner of input params created by createMetaStack. If function
  // returns reference to input param, dropping params cause memory leak.
  // In order to prevent it lifetime of inputs must be extedned.
  const c10::Stack input_tensor_liftetime_extender =
      copyTensorsFrom(meta_stack);

  ERROR_ON_MSG(!op.hasComputedKernelForDispatchKey(c10::DispatchKey::Meta),
               "Type inference failed for "
                   << schema_key
                   << " because the operator "
                      "doesn't have an implementation for the Meta backend.");
  logging::trace("[DISPATCHER] Using meta type inference for {}", schema_key);

  op.redispatchBoxed(meta_keys, &meta_stack);
  ipu_stack->clear();

  repopulateIpuStack(*ipu_stack, meta_stack, is_prelu);
}

std::optional<TypeInferenceHandler::Workaround>
TypeInferenceHandler::workaroundLookup(const std::string &schema_key) {
  if (const auto &it = schema_to_workaround.find(schema_key);
      it != schema_to_workaround.cend()) {
    return it->second;
  }

  return std::nullopt;
}

c10::IValue TypeInferenceHandler::applyWorkaround(
    const TypeInferenceHandler::Workaround &workaround, std::size_t value_index,
    const c10::IValue &value, const c10::Stack &stack) {

  if (workaround.predicate_fn(value_index, value, stack)) {
    return workaround.transform_fn(value, stack);
  }

  return value;
}

namespace {

template <typename T>
c10::List<T> createMetaTensorList(const c10::List<T> &ipu_tensor_list,
                                  bool should_upcast_to_long) {
  c10::List<T> meta_tensor_list;
  std::function<T(const T &)> transform_fn;

  if constexpr (std::is_same_v<c10::optional<at::Tensor>, T>) {
    transform_fn = [=](const T &t) -> T {
      if (!t.has_value()) {
        return c10::nullopt;
      }
      return TypeInferenceHandler::toMeta(t.value(), should_upcast_to_long);
    };
  } else {
    transform_fn = [=](const T &t) -> T {
      return TypeInferenceHandler::toMeta(t, should_upcast_to_long);
    };
  }

  std::transform(ipu_tensor_list.begin(), ipu_tensor_list.end(),
                 std::back_inserter(meta_tensor_list), transform_fn);

  return meta_tensor_list;
}

c10::Device createMetaDevice(const c10::Device &device) {
  return device.is_ipu() ? c10::Device{at::kMeta} : device;
}

bool isUpcastRequired(const std::string &schema_key,
                      const std::size_t input_idx) {
  if (auto opt_upcast_arg =
          TypeInferenceHandler::indexArgToUpcast(schema_key)) {
    return opt_upcast_arg.value() == input_idx;
  }

  return false;
}

} // namespace

c10::Stack TypeInferenceHandler::createMetaStack(const c10::Stack &ipu_stack,
                                                 const std::string &schema_key,
                                                 bool is_prelu) {
  c10::Stack meta_stack;
  meta_stack.reserve(ipu_stack.size());
  const auto maybe_workaround = workaroundLookup(schema_key);

  std::transform(
      ipu_stack.cbegin(), ipu_stack.cend(), std::back_inserter(meta_stack),
      [&, input_idx = 0u](const c10::IValue &value) mutable -> c10::IValue {
        // For various reasons, sometimes we have to transform the value before
        // pushing it on the meta stack to workaround validation issues which
        // are not the problem for the PopArt backend.
        const auto &v = maybe_workaround
                            ? applyWorkaround(maybe_workaround.value(),
                                              input_idx, value, ipu_stack)
                            : value;

        // We coerce index tensor types from Long to Int during dispatch, but
        // these need to be converted back to Long before running with the Meta
        // backend otherwise they'll emit type errors
        const bool should_upcast_to_long =
            isUpcastRequired(schema_key, input_idx);
        const bool is_first_input = input_idx == 0;

        ++input_idx;
        // Convert any IPU tensors to meta tensors
        if (v.isTensor()) {
          return toMeta(v.toTensor(), should_upcast_to_long,
                        is_prelu && is_first_input);
        }
        if (v.isTensorList()) {
          return createMetaTensorList(v.toTensorList(), should_upcast_to_long);
        }
        if (v.isOptionalTensorList()) {
          return createMetaTensorList(v.toOptionalTensorList(),
                                      should_upcast_to_long);
        }
        if (v.isDevice()) {
          return createMetaDevice(v.toDevice());
        }

        return v;
      });

  return meta_stack;
}

at::Tensor TypeInferenceHandler::allocateTensor(const at::Tensor &meta_tensor,
                                                bool is_prelu) {
  auto sizes = meta_tensor.sizes();
  if (is_prelu && sizes.size() == 2 && sizes[1] == 1) {
    sizes = sizes.slice(1);
  }

  return _tensor_store->allocateTensor(sizes, meta_tensor.scalar_type());
}

c10::List<at::Tensor> TypeInferenceHandler::allocateTensorList(
    const c10::List<at::Tensor> &meta_tensor_list) {
  c10::List<at::Tensor> allocated_tensor_list;

  std::transform(meta_tensor_list.begin(), meta_tensor_list.end(),
                 std::back_inserter(allocated_tensor_list),
                 [this](const at::Tensor &tensor) {
                   return this->_tensor_store->allocateTensor(
                       tensor.sizes(), tensor.scalar_type());
                 });

  return allocated_tensor_list;
}

void TypeInferenceHandler::repopulateIpuStack(c10::Stack &ipu_stack,
                                              const c10::Stack &meta_stack,
                                              bool is_prelu) {
  ERROR_ON(!ipu_stack.empty());
  ipu_stack.reserve(meta_stack.size());

  std::transform(meta_stack.cbegin(), meta_stack.cend(),
                 std::back_inserter(ipu_stack),
                 [=](const auto &v) -> c10::IValue {
                   if (v.isTensor()) {
                     return allocateTensor(v.toTensor(), is_prelu);
                   }
                   if (v.isTensorList()) {
                     return allocateTensorList(v.toTensorList());
                   }
                   return v;
                 });
}

namespace {

std::vector<int64_t> getMetaTensorSize(const at::Tensor &tensor,
                                       bool is_prelu) {
  std::vector<int64_t> sizes = tensor.sizes().vec();
  if (is_prelu && sizes.size() == 1) {
    sizes.push_back(1);
  }

  return sizes;
}

c10::ScalarType getMetaTensorDtype(const at::Tensor &tensor,
                                   bool should_upcast_to_long) {
  const auto dtype = tensor.scalar_type();
  if (dtype == c10::ScalarType::Int && should_upcast_to_long) {
    return c10::ScalarType::Long;
  }

  return dtype;
}

at::Tensor createEmptyMetaTensor(const at::Tensor &tensor,
                                 bool should_upcast_to_long, bool is_prelu) {
  const auto dtype = getMetaTensorDtype(tensor, should_upcast_to_long);
  const std::vector<long> sizes = getMetaTensorSize(tensor, is_prelu);

  auto out = at::meta::empty(sizes, dtype);

  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    out.unsafeGetTensorImpl()->set_wrapped_number(true);
  }

  return out;
}

} // namespace

at::Tensor TypeInferenceHandler::toMeta(const at::Tensor &tensor,
                                        bool should_upcast_to_long,
                                        bool is_prelu) {
  if (!tensor.defined()) {
    return tensor;
  }
  if (!isIpuTensor(tensor)) {
    if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
      return tensor;
    }
    ERROR("Expected an IPU tensor but got tensor(device="
          << tensor.device() << ", shape=" << tensor.sizes()
          << ", dtype=" << tensor.scalar_type()
          << ").\nConstant tensors should be moved explicitly "
             "to the IPU, via cpu_tensor.to(\"ipu\").");
  }

  return createEmptyMetaTensor(tensor, should_upcast_to_long, is_prelu);
}

c10::optional<std::size_t>
TypeInferenceHandler::indexArgToUpcast(const std::string &schema_key) {

  if (schema_key == "aten::argmax.out" || schema_key == "aten::argmin.out") {
    return 3;
  }
  if (schema_key == "aten::gather" || schema_key == "aten::scatter.src" ||
      schema_key == "aten::scatter_.src" ||
      schema_key == "aten::scatter.value" ||
      schema_key == "aten::scatter.value_reduce" ||
      schema_key == "aten::scatter_.value" ||
      schema_key == "aten::scatter_.value_reduce" ||
      schema_key == "aten::scatter_add" || schema_key == "aten::scatter_add_" ||
      schema_key == "aten::scatter_reduce.two" ||
      schema_key == "aten::scatter_reduce_.two" ||
      schema_key == "torch_scatter::scatter_max" ||
      schema_key == "torch_scatter::scatter_min" ||
      schema_key == "torch_scatter::scatter_mul") {
    return 2;
  }
  if (schema_key == "aten::index.Tensor" ||
      schema_key == "aten::nll_loss_forward") {
    return 1;
  }
  if (schema_key == "aten::sort.values_stable") {
    return 5;
  }
  return c10::nullopt;
}

static bool reductionWorkaroundPredicate(const std::size_t value_index,
                                         const c10::IValue &value,
                                         const c10::Stack &ipu_stack,
                                         const std::size_t dtype_index,
                                         const std::size_t out_index) {

  return value_index == dtype_index && value.isNone() &&
         !ipu_stack.at(out_index).isNone();
}

static c10::IValue reductionTransform(const c10::IValue &transformed_value,
                                      const c10::Stack &ipu_stack,
                                      const std::size_t out_index) {
  const auto &value = ipu_stack.at(out_index);
  if (!value.isNone() && value.isTensor()) {
    const auto tensor = value.toTensor();
    return c10::IValue(c10::typeMetaToScalarType(tensor.dtype()));
  }
  return transformed_value;
}

static auto makeReductionWorkaround(const std::size_t dtype_index,
                                    const std::size_t out_index) {

  /* In case dtype is None, PyTorch meta backend assumes that it is int64_t for
   * all integral tensors, causing validation issues when the output tensor has
   * int32_t dtype.
   */

  const auto predicate = [=](const std::size_t value_index,
                             const c10::IValue &value,
                             const c10::Stack &ipu_stack) {
    return reductionWorkaroundPredicate(value_index, value, ipu_stack,
                                        dtype_index, out_index);
  };

  const auto transform_fn = [=](const c10::IValue &transformed_value,
                                const c10::Stack &ipu_stack) {
    return reductionTransform(transformed_value, ipu_stack, out_index);
  };

  return TypeInferenceHandler::Workaround{predicate, transform_fn};
}
const std::unordered_map<std::string, TypeInferenceHandler::Workaround>
    TypeInferenceHandler::schema_to_workaround = {
        {"aten::sum.IntList_out",
         makeReductionWorkaround(3 /*dtype_index*/, 4 /*out_index*/)},
        {"aten::cumsum.out",
         makeReductionWorkaround(2 /*dtype_index*/, 3 /*out_index*/)},
        {"aten::cumprod.out",
         makeReductionWorkaround(2 /*dtype_index*/, 3 /*out_index*/)},
        {"aten::sum.out",
         makeReductionWorkaround(4 /*dtype_index*/, 0 /*out_index*/)},
        {"aten::prod.out",
         makeReductionWorkaround(4 /*dtype_index*/, 0 /*out_index*/)}};
} // namespace poptorch
