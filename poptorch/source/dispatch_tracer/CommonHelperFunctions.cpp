// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CommonHelperFunctions.hpp"

#include <torch/csrc/jit/ir/ir.h>

#include <map>
#include <unordered_set>

#include "../popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "poptorch/DispatchTracer.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Logging.hpp"

#include "ValueMapper.hpp"

namespace poptorch {

namespace {

bool isGenericListOfTensors(c10::IValue &value) {
  if (!value.isList()) {
    return false;
  }
  bool not_empty = false;
  for (c10::IValue list_value : value.toList()) {
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
    at::Tensor tensor = value.toTensor();
    // Undefined tensors are optional tensors.
    if (!tensor.defined()) {
      // Create a null IR value.
      torch::jit::Node *node = graph.createNone();
      insertNodeInGraph(&graph, node);
      return node->output();
    }

    torch::jit::Value *val = mapper.getValueForTensor(tensor);
    if (val == nullptr) {
      // This is probably an external tensor that we didn't catch. Assume
      // it's a constant.
      val = makeConstant(graph, tensor);
      // Don't track constants in the ValueMapper as they are CPU tensors.
    }

    logging::trace(
        "[TRACING-2] Tensor input: tensor ptr {} ({}), jit ir %{} (scalar type "
        "{})",
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
  torch::jit::Value *val = graph.insertConstant(value);
  ERROR_ON_MSG(val == nullptr, "Internal: graph could not insert a constant");

  logging::trace("[TRACING-2] Constant input: jit ir %{}, ivalue tag kind: {}",
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
  torch::jit::Symbol symbol = torch::jit::Symbol::fromQualString(schema.name());

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
        at::Tensor as_tensor = val.toTensor();
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

torch::jit::Value *makeConstant(torch::jit::Graph &graph,
                                const at::Tensor &tensor) {
  auto *constant = tensorToConstant(&graph, copyAndCoerceType(tensor));
  return constant->output();
}

at::Tensor copyAndCoerceType(const at::Tensor &tensor) {
  at::Tensor copy;
  auto scalar_type = tensor.scalar_type();
  auto coerced_scalar_type = coerceToSupportedType(scalar_type);
  if (scalar_type != coerced_scalar_type) {
    logging::warn("[TRACING-2] Tensor (ptr {}) type coerced from {} to {}",
                  static_cast<void *>(tensor.unsafeGetTensorImpl()),
                  scalar_type, coerced_scalar_type);
    return tensor.to(coerced_scalar_type);
  }
  return tensor;
}

c10::OperatorHandle getOutplaceOpHandle(const c10::OperatorHandle &op,
                                        c10::Dispatcher &dispatcher,
                                        c10::Stack &stack) {
  const auto &schema = op.schema();
  std::string name = schema.name();
  const std::string &overload = schema.overload_name();
  // If ends with '_', it's inplace. Remove the "_" and use the outplace version
  // instead.
  if (name[name.size() - 1] == '_') {
    // These are special cases because there is no zero / fill.
    if (name == "aten::zero_") {
      name = "aten::zeros_like";
      // zero_ takes only 1 argument whereas our MLIR shape inference for
      // zeros_like takes an optional dtype as well
      stack.emplace_back();
    } else if (name == "aten::fill_") {
      name = "aten::full_like";
    } else {
      name.erase(name.end() - 1, name.end());
    }
    auto opt_op = dispatcher.findOp({name, overload});
    if (opt_op) {
      return *opt_op;
    }
    opt_op = dispatcher.findOp({name, ""});
    if (opt_op) {
      return *opt_op;
    }
  }
  return op;
}

std::optional<at::Tensor>
getInplaceArgument(const c10::Stack &stack, const c10::FunctionSchema &schema) {
  logging::trace("[TRACING-2][JIT] Looking for inplace argument in schema {}",
                 schema);

  for (std::size_t arg = 0; arg < schema.arguments().size(); ++arg) {
    const c10::Argument &argument = schema.arguments()[arg];
    c10::IValue value = stack[arg];

    if (value.isTensor()) {
      at::Tensor tensor = value.toTensor();

      // Undefined tensors are optional tensors.
      if (!tensor.defined()) {
        continue;
      }

      if (argument.alias_info() && argument.alias_info()->isWrite()) {
        // We just return the first inplace argument but more than one can
        // technically be inplace.
        logging::trace(
            "[TRACING-2][JIT] Found inplace argument, tensor ptr {}, tensor {}",
            reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
            toString(tensor));
        return tensor;
      }
    }
  }

  // Assigned null in constructor.
  return std::nullopt;
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

void assertCanBeCanonicalised(const c10::FunctionSchema &schema) {
  torch::jit::Symbol symbol = torch::jit::Symbol::fromQualString(schema.name());
  if (schema.getNamespace().has_value() &&
      *schema.getNamespace() == "poptorch") {
    // OK: it's a PopTorch op (It will be handled by the late canonicalisation).
    return;
  }
  // In the JIT path we are not allowed to fail as we only have the
  // canonicaliser to rely on. In the MLIR path we have our own 1:1 handlers
  // as well so we can use them too and we will only fail if BOTH JIT and MLIR
  // can't process the node.
  ERROR_ON_MSG(!getHandler(symbol),
               "Could not find canonicalisation handler for JIT symbol, "
                   << symbol.toQualString() << ", for schema, " << schema.name()
                   << ".");
}

std::string toString(const at::Tensor &t) {
  std::stringstream ss;
  ss << "sizes=" << t.sizes() << ", type=" << t.scalar_type();
  return ss.str();
}

bool isHalfTensor(const at::Tensor &t) {
  return t.scalar_type() == at::ScalarType::Half;
}

c10::Device deviceOrDefaultIpu(c10::optional<c10::Device> device) {
  // TODO(T59880) rename kXLA -> kIPU
  return device ? *device : c10::Device(at::kXLA, 0);
}

} // namespace poptorch
