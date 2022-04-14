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

torch::jit::Value *insertValueIntoGraphAndTrackIt(c10::IValue &value,
                                                  torch::jit::Graph &graph,
                                                  ValueMapper &mapper) {
  if (value.isTensor()) {
    // Handle tensors.
    at::Tensor tensor = value.toTensor();
    // Undefined tensors are optional tensors.
    if (!tensor.defined()) {
      // Create a null IR value.
      torch::jit::Node *node = graph.insertNode(graph.createNone());
      return node->output();
    }

    torch::jit::Value *val = mapper.getValueForTensor(tensor);
    if (val == nullptr) {
      // This is probably an external tensor that we didn't catch. Assume
      // it's a constant.
      val = makeConstant(graph, tensor);
      mapper.addTensor(tensor, val);
    } else {
      auto *record = mapper.rawTensorRecord(tensor);
      // If this isn't an input or node-output tensor, add it to the graph now
      // as a constant and fix the mapping.
      if (record->is_empty) {
        val = makeConstant(graph, tensor);
        mapper.addTensor(tensor, val);
      }
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

  if (value.isTensorList()) {
    // Handle tensor lists.
    std::vector<torch::jit::Value *> list_values;
    for (c10::IValue list_value : value.toTensorVector()) {
      list_values.push_back(
          insertValueIntoGraphAndTrackIt(list_value, graph, mapper));
    }

    // We assume all lists with the same jit values are the same list in python.
    torch::jit::Value *val = mapper.getValueForTensorList(list_values);
    if (val == nullptr) {
      auto *list = graph.createList(c10::TensorType::get(), list_values);
      graph.insertNode(list);
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
      graph.create(symbol, inputs, schema.returns().size());
  graph.insertNode(aten_target);

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

  poptorch::type_and_constant_canonicalization::canonicaliseConstantsDispatch(
      &graph, aten_target);

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
    copy = tensor.to(coerced_scalar_type);
  } else {
    copy = tensor.clone();
  }
  return copy;
}

c10::OperatorHandle getOutplaceOpHandle(const c10::OperatorHandle &op,
                                        c10::Dispatcher &dispatcher) {
  const auto &schema = op.schema();
  std::string name = schema.name();
  const std::string &overload = schema.overload_name();
  // If ends with '_', it's inplace. Remove the "_" and use the outplace version
  // instead.
  if (name[name.size() - 1] == '_') {
    // These are special cases because there is no zero / fill.
    if (name == "aten::zero_") {
      name = "aten::zeros_like";
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

c10::intrusive_ptr<at::TensorImpl>
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
        return tensor.getIntrusivePtr();
      }
    }
  }

  // Assigned null in constructor.
  return {};
}

torch::jit::Node *lowerFromSchema(const c10::FunctionSchema &schema,
                                  c10::Stack *stack, torch::jit::Graph &graph,
                                  ValueMapper &mapper) {
  std::vector<torch::jit::Value *> inputs;
  for (std::size_t arg = 0;
       arg < schema.arguments().size() && arg < stack->size(); ++arg) {
    auto value = (*stack)[arg];
    const auto &argument = schema.arguments()[arg];
    bool is_write = argument.alias_info() && argument.alias_info()->isWrite();
    // If we're writing to this tensor, it's no longer empty
    if (is_write && value.isTensor()) {
      at::Tensor tensor = value.toTensor();
      // Undefined tensors are optional tensors.
      if (tensor.defined()) {
        auto *record = mapper.rawTensorRecord(tensor);
        if (record != nullptr) {
          record->is_empty = false;
        }
      }
    }
    inputs.push_back(insertValueIntoGraphAndTrackIt(value, graph, mapper));
  }
  return createAtenTarget(graph, schema, inputs, stack, mapper);
}

void fixNodeOutput(torch::jit::Node *node, const c10::Stack &stack,
                   ValueMapper &mapper) {
  std::uint32_t index = 0;
  for (c10::IValue value : stack) {
    // Add any missing outputs. They frequently return scalars which we just
    // ignore here as our canonicalisation only returns tensors.
    while (index >= node->outputs().size()) {
      node->addOutput();
    }

    if (value.isTensor()) {
      at::Tensor tensor = value.toTensor();
      // Sometimes "Tensors" are actually "Not tensors" but still stored as a
      // tensor and will assert in the infer type.
      if (tensor.sizes().size() == 1 && tensor.sizes()[0] == 0) {
        continue;
      }

      if (mapper.isHalfTensor(tensor)) {
        tensor = tensor.toType(at::ScalarType::Half);
      }
      torch::jit::Value *val = node->output(index);
      val->inferTypeFrom(tensor);
    } else if (value.isTensorList()) {
      auto list_type = value.type()->expect<c10::ListType>();
      torch::jit::Value *val = node->output(index);
      val->setType(list_type);
    }
    index++;
  }
}

static std::map<torch::jit::Value *, torch::jit::Value *> replacements;

torch::jit::Node *canonicalise(const c10::FunctionSchema &schema,
                               torch::jit::Node *aten_target,
                               torch::jit::Graph &graph,
                               bool is_allowed_to_fail) {
  replacements.clear();
  // Run the normal canonicalisation process on the aten target.
  torch::jit::Symbol symbol = torch::jit::Symbol::fromQualString(schema.name());

  torch::jit::Node *new_node = nullptr;

  if (SymbolHandler handler = getHandler(symbol)) {
    new_node = handler(&graph, aten_target);
    // No new node: keep the existing one.
    if (new_node == nullptr) {
      new_node = aten_target;
    } else {
      // If we have a new node add it and replace the old use.
      std::unordered_set<torch::jit::Node *> to_delete;
      to_delete.insert(aten_target);

      // Clean up any dead nodes.
      searchAndPossiblyDestroy(to_delete);
    }
  } else {
    // In the JIT path we are not allowed to fail as we only have the
    // canonicaliser to rely on. In the MLIR path we have our own 1:1 handlers
    // as well so we can use them too and we will only fail if BOTH JIT and MLIR
    // can't process the node.
    ERROR_ON_MSG(!is_allowed_to_fail,
                 "Could not find canonicalisation handler for JIT symbol: "
                     << symbol.toQualString());
    new_node = aten_target;
  }
  return new_node;
}

torch::jit::Value *wasReplaced(torch::jit::Value *target) {
  auto it = replacements.find(target);
  if (it == std::end(replacements)) {
    return nullptr;
  }
  return it->second;
}

std::string toString(const at::Tensor &t) {
  std::stringstream ss;
  ss << "sizes=" << t.sizes() << ", type=" << t.scalar_type();
  return ss.str();
}

void replaceAllUsesWith(torch::jit::Value *target,
                        torch::jit::Value *replacement) {
  if (isDispatcherActive()) {
    replacements[target] = replacement;
  }
  target->replaceAllUsesWith(replacement);
}

void replaceAllUsesAfterNodeWith(torch::jit::Node *node,
                                 torch::jit::Value *target,
                                 torch::jit::Value *replacement) {
  if (isDispatcherActive()) {
    replacements[target] = replacement;
  }
  target->replaceAllUsesAfterNodeWith(node, replacement);
}

bool isHalfTensor(const at::Tensor &t) {
  return t.scalar_type() == at::ScalarType::Half;
}

} // namespace poptorch
