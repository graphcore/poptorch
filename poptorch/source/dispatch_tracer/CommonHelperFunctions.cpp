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

torch::jit::Value *makeConstant(torch::jit::Graph &graph, at::Tensor &tensor) {
  at::Tensor ct;
  auto st = tensor.scalar_type();
  auto st_coerced = coerceToSupportedType(st);
  if (st != st_coerced) {
    ct = tensor.to(st_coerced);
  } else {
    ct = tensor.clone();
  }
  auto *constant = tensorToConstant(&graph, ct);
  return constant->output();
}

torch::jit::Value *insertValueIntoGraph(c10::IValue &value,
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
    // If we couldn't find the tensor in the tensors we are currently tracking.
    if (val == nullptr) {
      // If it is actually just a tensor wrapper around a python scalar we just
      // add it as a constant.
      if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
        val = graph.insertConstant(value);
      } else {
        // This is probably an external tensor that we didn't catch. Assume
        // it's a constant.
        val = makeConstant(graph, tensor);
      }
    } else {
      // If this is a constant tensor, add it to the graph now
      auto is_const = *mapper.tensorIsConst(tensor);
      if (is_const) {
        val = makeConstant(graph, tensor);
      }
    }

    logging::trace("[TRACING-2] Input: Tensor ptr {}, jit ir {} {}",
                   reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
                   val->debugNameBase(), toString(tensor));

    return val;
  }

  if (value.isTensorList()) {
    // Handle tensor lists.
    std::vector<torch::jit::Value *> list_values;
    for (c10::IValue list_value : value.toTensorVector()) {
      list_values.push_back(insertValueIntoGraph(list_value, graph, mapper));
    }
    auto *list = graph.createList(c10::TensorType::get(), list_values);
    graph.insertNode(list);
    return list->output();
  }

  // Assume value is a constant.
  torch::jit::Value *val = graph.insertConstant(value);
  ERROR_ON_MSG(val == nullptr, "Internal: graph could not insert a constant");
  return val;
}

} // namespace

// Create the aten target node.
// Note: Since 1.9.0 we could also call:
// aten_target = aten_target->replaceWithNewSymbol(symbol);
// To swap out the symbols.
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
        // to constant.
        torch::jit::Value *new_val = mapper.getValueForTensor(as_tensor);

        if ((new_val != nullptr) && new_val != in) {
          in->replaceAllUsesWith(new_val);
          in->node()->destroy();
        }
      }
    }
  }

  poptorch::type_and_constant_canonicalization::canonicaliseConstants(&graph);
  poptorch::resolveHalfOrFloat(&graph);

  return aten_target;
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

torch::jit::Value *wasReplaced(torch::jit::Value *target) {
  auto it = replacements.find(target);
  if (it == std::end(replacements)) {
    return nullptr;
  }
  return it->second;
}

// From the given torch schema return the correct mlir values.
torch::jit::Node *lowerFromSchema(const c10::FunctionSchema &schema,
                                  c10::Stack *stack, torch::jit::Graph &graph,
                                  ValueMapper &mapper) {
  std::vector<torch::jit::Value *> inputs;
  for (c10::IValue value : *stack) {
    inputs.push_back(insertValueIntoGraph(value, graph, mapper));
  }
  return createAtenTarget(graph, schema, inputs, stack, mapper);
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

bool isTrulyInplace(const c10::Stack &stack,
                    const c10::FunctionSchema &schema) {
  const auto &arguments = schema.arguments();
  for (size_t i = 0; i < arguments.size(); ++i) {
    const auto &argument = arguments.at(i);
    const c10::IValue value = stack.at(i);

    // We are only interested in finding inplace tensors.
    if (!argument.alias_info() || !argument.alias_info()->isWrite() ||
        !value.isTensor()) {
      continue;
    }
    const at::Tensor &tensor = value.toTensor();
    // Undefined tensors are optional tensors.
    if (!tensor.defined()) {
      continue;
    }

    // We've found an inplace tensor. Check the stack whether it actually
    // references one of the input arguments and it isn't just
    // the 'out' argument.
    for (size_t j = 0; j < stack.size(); ++j) {
      if (i == j) {
        continue;
      }
      if (value.isSameIdentity(stack.at(j))) {
        return true;
      }
    }
  }

  return false;
}

std::string toString(const at::Tensor &t) {
  std::stringstream ss;
  ss << "sizes=" << t.sizes() << ", type=" << t.scalar_type();
  return ss.str();
}

} // namespace poptorch
