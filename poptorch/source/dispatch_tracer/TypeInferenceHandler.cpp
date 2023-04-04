// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <ATen/MetaFunctions.h>

#include "poptorch_logging/Logging.hpp"

#include "CommonHelperFunctions.hpp"
#include "TypeInferenceHandler.hpp"

namespace poptorch {

constexpr c10::DispatchKeySet meta_keys{c10::DispatchKey::Meta,
                                        c10::DispatchKey::AutogradMeta};

void TypeInferenceHandler::inferOutputTypes(const c10::OperatorHandle &op,
                                            c10::Stack *ipu_stack) {
  auto schema_key = getSchemaKey(op.schema());
  // Unfortunately, aten::prelu with 1D inputs is broken with the Meta backend:
  // https://github.com/pytorch/pytorch/issues/89560
  // As a workaround, we add a dummy channel dim to the input, and then remove
  // it again afterwards
  const bool is_prelu =
      schema_key == "aten::prelu" || schema_key == "aten::_prelu_kernel";
  // Create a new operand stack with meta tensors
  c10::Stack meta_stack = createMetaStack(*ipu_stack, schema_key, is_prelu);

  ERROR_ON_MSG(!op.hasComputedKernelForDispatchKey(c10::DispatchKey::Meta),
               "Type inference failed for "
                   << schema_key
                   << " because the operator "
                      "doesn't have an implementation for the Meta backend.");
  logging::trace("[DISPATCHER] Using meta type inference for {}", schema_key);

  op.redispatchBoxed(meta_keys, &meta_stack);

  ipu_stack->clear();
  repopulateIpuStack(ipu_stack, meta_stack, is_prelu);
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

c10::Stack TypeInferenceHandler::createMetaStack(const c10::Stack &ipu_stack,
                                                 const std::string &schema_key,
                                                 bool is_prelu) {
  c10::Stack meta_stack;

  const auto maybe_workaround = workaroundLookup(schema_key);

  // Create meta tensors and add them to the meta stack
  for (auto i = 0u; i < ipu_stack.size(); i++) {
    // For various reasons, sometimes we have to transform the value before
    // pushing it on the meta stack to workaround validation issues which are
    // not the problem for the PopArt backend.
    const auto &v = maybe_workaround
                        ? applyWorkaround(maybe_workaround.value(), i,
                                          ipu_stack[i], ipu_stack)
                        : ipu_stack[i];

    // We coerce index tensor types from Long to Int during dispatch, but these
    // need to be converted back to Long before running with the Meta backend
    // otherwise they'll emit type errors
    bool should_upcast_to_long = false;

    if (auto opt_upcast_arg = indexArgToUpcast(schema_key)) {
      should_upcast_to_long = opt_upcast_arg.value() == i;
    }
    // Convert any IPU tensors to meta tensors
    if (v.isTensor()) {
      meta_stack.push_back(
          toMeta(v.toTensor(), should_upcast_to_long, is_prelu && i == 0));
    } else if (v.isTensorList()) {
      const c10::List<at::Tensor> l;
      for (const at::Tensor &t : v.toTensorList()) {
        l.push_back(toMeta(t, should_upcast_to_long));
      }
      meta_stack.push_back(l);
    } else if (v.isOptionalTensorList()) {
      const c10::List<c10::optional<at::Tensor>> l;
      for (c10::optional<at::Tensor> o : v.toOptionalTensorList()) {
        if (!o.has_value()) {
          l.push_back(c10::nullopt);
          continue;
        }
        l.push_back(toMeta(o.value(), should_upcast_to_long));
      }
      meta_stack.push_back(l);
    } else if (v.isDevice()) {
      auto d = v.toDevice();
      if (d.is_ipu()) {
        d = c10::Device{at::kMeta};
      }
      meta_stack.push_back(d);
    } else {
      meta_stack.push_back(v);
    }
  }
  return meta_stack;
}

void TypeInferenceHandler::repopulateIpuStack(c10::Stack *ipu_stack,
                                              const c10::Stack &meta_stack,
                                              bool is_prelu) {
  ERROR_ON(!ipu_stack->empty());
  for (const auto &v : meta_stack) {
    if (v.isTensor()) {
      const auto &t = v.toTensor();
      auto sizes = t.sizes();
      if (is_prelu && sizes.size() == 2 && sizes[1] == 1) {
        sizes = sizes.slice(1);
      }
      auto t1 = _tensor_store->allocateTensor(sizes, t.scalar_type());
      ipu_stack->push_back(t1);
    } else if (v.isTensorList()) {
      const c10::List<at::Tensor> l;
      for (const at::Tensor &t : v.toTensorList()) {
        auto t1 = _tensor_store->allocateTensor(t.sizes(), t.scalar_type());
        l.push_back(t1);
      }
      ipu_stack->push_back(l);
    } else {
      ipu_stack->push_back(v);
    }
  }
}

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
  auto dtype = tensor.scalar_type();
  if (dtype == c10::ScalarType::Int && should_upcast_to_long) {
    dtype = c10::ScalarType::Long;
  }
  auto sizes = tensor.sizes().vec();
  if (is_prelu && sizes.size() == 1) {
    sizes.push_back(1);
  }
  auto out = at::meta::empty(sizes, dtype);
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    out.unsafeGetTensorImpl()->set_wrapped_number(true);
  }
  return out;
}

c10::optional<std::size_t>
TypeInferenceHandler::indexArgToUpcast(const std::string &schema_key) {

  if (schema_key == "aten::argmax.out" || schema_key == "aten::argmin.out") {
    return 3;
  }
  if (schema_key == "aten::gather" || schema_key == "aten::scatter.src" ||
      schema_key == "aten::scatter_.src" ||
      schema_key == "aten::scatter.value" ||
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
