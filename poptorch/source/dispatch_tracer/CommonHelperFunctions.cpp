// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CommonHelperFunctions.hpp"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#include <ATen/core/function_schema.h>
#include <ATen/core/interned_strings.h>
#include <c10/core/TensorImpl.h>
#include <torch/csrc/jit/ir/ir.h>

#include <map>
#include <unordered_set>

#include "../popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "InplaceAliasMapper.hpp"
#include "ValueMapper.hpp"
#include "poptorch/DispatchTracer.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

namespace {

bool isGenericListOfTensors(c10::IValue &value) {
  if (!value.isList()) {
    return false;
  }
  bool not_empty = false;
  for (c10::IValue const list_value : value.toList()) {
    if (!list_value.isTensor()) {
      return false;
    }
    not_empty = true;
  }
  return not_empty;
}

bool isListOfOptionalTensors(c10::IValue &value) {
  if (!value.isList()) {
    return false;
  }
  return value.toList().elementType() ==
         c10::getTypePtr<c10::optional<at::Tensor>>();
}

torch::jit::Value *insertValueIntoGraphAndTrackIt(c10::IValue &value,
                                                  torch::jit::Graph &graph,
                                                  ValueMapper &mapper) {
  if (value.isTensor()) {
    // Handle tensors.
    at::Tensor const tensor = value.toTensor();
    // Undefined tensors are optional tensors.
    if (!tensor.defined()) {
      // Create a null IR value.
      torch::jit::Node *node = graph.createNone();
      insertNodeInGraph(&graph, node);
      return node->output();
    }

    torch::jit::Value *val = mapper.getValueForTensor(tensor);
    if (val == nullptr) {
      ERROR_ON_MSG(tensor.device().type() == c10::DeviceType::IPU,
                   "Attempted to promote a Tensor converted (using "
                   ".to(\"ipu\") or .ipu()) outside an IPUScope or IPUContext "
                   "with the PopART compiler.");

      // This is probably an external tensor that we didn't catch. Assume
      // it's a constant.
      val = insertConstant(graph, copyAndCoerceType(tensor));
      setSourceRangeToCurrentLocation(val->node());
      // Don't track constants in the ValueMapper as they are CPU tensors.
    }

    logging::trace(
        "[DISPATCHER] Tensor input: tensor ptr {} ({}), jit ir %{} (scalar "
        "type {})",
        reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
        toString(tensor), val->debugNameBase(),
        val->type()->expect<c10::TensorType>()->scalarType().value_or(
            at::ScalarType::Undefined));

    return val;
  }

  // If a generic list only contains tensors then it is a tensor
  // list and we handle both the same way.
  if (value.isTensorList() || isGenericListOfTensors(value)) {
    // Handle tensor lists.
    std::vector<torch::jit::Value *> list_values;
    if (value.isTensorList()) {
      for (c10::IValue list_value : value.toTensorVector()) {
        list_values.push_back(
            insertValueIntoGraphAndTrackIt(list_value, graph, mapper));
      }
    } else {
      for (c10::IValue list_value : value.toList()) {
        list_values.push_back(
            insertValueIntoGraphAndTrackIt(list_value, graph, mapper));
      }
    }

    // We assume all lists with the same jit values are the same list in python.
    torch::jit::Value *val = mapper.getValueForTensorList(list_values);
    if (val == nullptr) {
      c10::TypePtr type_ptr;
      if (value.isTensorList()) {
        type_ptr = c10::TensorType::get();
      } else if (isListOfOptionalTensors(value)) {
        type_ptr = c10::OptionalType::create(c10::TensorType::get());
      }

      auto *list = graph.createList(type_ptr, list_values);
      insertNodeInGraph(&graph, list);
      val = list->output();
      mapper.addTensorList(list_values, val);
    }
    return val;
  }

  // Assume value is a true constant and not a tensor so we don't have to
  // track it in the value mapper. It will get canonicalised later.
  torch::jit::Value *val = insertConstant(&graph, value);
  ERROR_ON_MSG(val == nullptr, "Internal: graph could not insert a constant");

  logging::trace("[DISPATCHER] Constant input: jit ir %{}, ivalue tag kind: {}",
                 val->debugNameBase(), value.tagKind());

  return val;
}

// Create a node based on the schema which deduces the input types
// from the inputs/stack and the name from the schema. As far as our
// canonicalisation is concerned this *is* the "aten" node it purports to be
// however it may not match it exacty, and is not created by the normal JIT
// process.
torch::jit::Node *
createAtenTarget(torch::jit::Graph &graph, const c10::FunctionSchema &schema,
                 const std::vector<torch::jit::Value *> &inputs,
                 c10::Stack *stack, ValueMapper &mapper) {

  logging::trace("[DISPATCHER] Create aten target {}", schema.name());

  torch::jit::Symbol const symbol =
      torch::jit::Symbol::fromQualString(schema.name());

  // Create the aten target node for our canonicalisation to target.
  torch::jit::Node *aten_target =
      createAndInsertNode(&graph, symbol, inputs, ImplicitCast::None,
                          OutputType::Unknown, schema.returns().size());

  for (std::size_t i = 0; i < aten_target->inputs().size(); ++i) {
    torch::jit::Value *in = aten_target->input(i);
    // If we are a constant.
    if (in->node()->kind() == at::prim::Constant) {
      c10::IValue val = stack->at(i);
      if (val.isTensor()) {
        at::Tensor const as_tensor = val.toTensor();
        // But actually we are a previously seen tensor which has been demoted
        // to a constant.
        torch::jit::Value *new_val = mapper.getValueForTensor(as_tensor);

        if ((new_val != nullptr) && new_val != in) {
          in->replaceAllUsesWith(new_val);
          in->node()->destroy();
        }
      }
    }
  }

  return aten_target;
}

} // namespace

at::ScalarType scalarTypeOrDefault(c10::optional<at::ScalarType> dtype) {
  return dtype ? *dtype : at::ScalarType::Float;
}

at::Tensor copyAndCoerceType(const at::Tensor &tensor) {
  at::Tensor const copy;
  const auto scalar_type = tensor.scalar_type();
  const auto coerced_scalar_type = coerceToSupportedType(scalar_type);
  if (scalar_type != coerced_scalar_type) {
    static std::uint64_t log_repeat = 0;
    logging::warn(log_repeat,
                  "[DISPATCHER] Tensor (ptr {}) type coerced from {} to {}",
                  static_cast<void *>(tensor.unsafeGetTensorImpl()),
                  scalar_type, coerced_scalar_type);
    return tensor.to(coerced_scalar_type);
  }
  return tensor;
}

std::vector<at::Tensor> getInplaceArguments(const c10::Stack &stack,
                                            const c10::FunctionSchema &schema) {
  logging::trace("[DISPATCHER][JIT] Looking for inplace arguments in schema {}",
                 schema);

  std::vector<at::Tensor> results;

  const auto inplace_arg_id =
      InplaceArgAliasMapper::getInplaceArg(schema.name());

  for (std::size_t arg = 0; arg < schema.arguments().size(); ++arg) {
    const c10::Argument &argument = schema.arguments()[arg];
    const c10::IValue value = stack[arg];

    if (value.isTensor()) {
      at::Tensor const &tensor = value.toTensor();

      // Undefined tensors are optional tensors.
      if (!tensor.defined()) {
        continue;
      }

      if (((argument.alias_info() != nullptr) &&
           argument.alias_info()->isWrite()) ||
          inplace_arg_id == arg) {
        logging::trace("[DISPATCHER][JIT] Found inplace argument, tensor ptr "
                       "{}, tensor {}",
                       reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
                       toString(tensor));
        results.push_back(tensor);
      }
    }
  }

  return results;
}

torch::jit::Node *lowerFromSchema(const c10::FunctionSchema &schema,
                                  c10::Stack *stack, torch::jit::Graph &graph,
                                  ValueMapper &mapper) {
  std::vector<torch::jit::Value *> inputs;
  for (std::size_t arg = 0;
       arg < schema.arguments().size() && arg < stack->size(); ++arg) {
    auto value = (*stack)[arg];
    inputs.push_back(insertValueIntoGraphAndTrackIt(value, graph, mapper));
  }
  return createAtenTarget(graph, schema, inputs, stack, mapper);
}

std::string toString(const at::Tensor &t) {
  return fmt::format("sizes={}, type={}", t.sizes(), t.scalar_type());
}

bool isHalfTensor(const at::Tensor &t) {
  return t.scalar_type() == at::ScalarType::Half;
}

c10::Device deviceOrDefaultIpu(c10::optional<c10::Device> device) {
  return device ? *device : c10::Device(at::kIPU, 0);
}

std::string getSchemaKey(const c10::FunctionSchema &schema) {
  // Unfortunately we can't overload based only on the schema symbol as it does
  // not contain the overload info.
  if (schema.overload_name().empty()) {
    return schema.name();
  }

  return schema.name() + "." + schema.overload_name();
}

} // namespace poptorch
