// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CommonHelperFunctions.hpp"

#include <torch/csrc/jit/ir/ir.h>

#include <unordered_set>

#include "../popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Logging.hpp"

#include "ValueMapper.hpp"

namespace poptorch {

namespace {

torch::jit::Value *trackValue(c10::IValue &value, torch::jit::Graph &graph,
                              ValueMapper &mapper) {
  if (value.isTensor()) {
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
        auto *constant = tensorToConstant(&graph, tensor.clone());
        val = constant->output();
      }
    } else {
      // If this is a constant tensor, add it to the graph now
      auto is_const = *mapper.tensorIsConst(tensor);
      if (is_const) {
        auto *constant = tensorToConstant(&graph, tensor.clone());
        val = constant->output();
      }
    }

    logging::trace("[TRACING-2] Input: Tensor ptr {}, jit ir {} {}",
                   reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
                   val->debugNameBase(), toString(tensor));
    return val;
  }

  torch::jit::Value *val = graph.insertConstant(value);

  ERROR_ON_MSG(val == nullptr, "Graph could not insert constant");
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

torch::jit::Node *canonicalise(const c10::FunctionSchema &schema,
                               torch::jit::Node *aten_target,
                               torch::jit::Graph &graph,
                               bool is_allowed_to_fail) {
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

// From the given torch schema return the correct mlir values.
torch::jit::Node *lowerFromSchema(const c10::FunctionSchema &schema,
                                  c10::Stack *stack, torch::jit::Graph &graph,
                                  ValueMapper &mapper) {
  std::vector<torch::jit::Value *> inputs;

  for (c10::IValue value : *stack) {
    torch::jit::Value *jit_value;

    if (value.isTensorList()) {
      std::vector<torch::jit::Value *> values;
      for (c10::IValue list_val : value.toTensorVector()) {
        values.push_back(trackValue(list_val, graph, mapper));
      }

      jit_value = graph.createList(c10::TensorType::get(), values)->output();
    } else {
      jit_value = trackValue(value, graph, mapper);
    }

    inputs.push_back(jit_value);
  }

  return createAtenTarget(graph, schema, inputs, stack, mapper);
}

c10::intrusive_ptr<at::TensorImpl>
getInplaceArgument(c10::Stack &stack, const c10::FunctionSchema &schema) {
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
        return tensor.getIntrusivePtr();
      }
    }
  }

  // Assigned null in constructor.
  return {};
}

std::string toString(const at::Tensor &t) {
  std::stringstream ss;
  ss << "sizes=" << t.sizes() << ", type=" << t.scalar_type();
  return ss.str();
}

} // namespace poptorch
