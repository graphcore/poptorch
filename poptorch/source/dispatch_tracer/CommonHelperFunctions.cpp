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
  return aten_target;
}

} // namespace

bool getOutplaceOpHandle(c10::OperatorHandle &op, c10::Dispatcher &dispatcher) {
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
      op = *opt_op;
    } else {
      op = *dispatcher.findOp({name, ""});
    }
    return true;
  }
  return false;
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

torch::jit::Node *lowerFromSchema(const c10::FunctionSchema &schema,
                                  c10::Stack *stack, torch::jit::Graph &graph,
                                  ValueMapper &mapper) {
  std::vector<torch::jit::Value *> inputs;
  for (std::size_t arg = 0;
       arg < schema.arguments().size() && arg < stack->size(); ++arg) {
    auto value = (*stack)[arg];
    const auto &argument = schema.arguments()[arg];
    bool is_write = argument.alias_info() && argument.alias_info()->isWrite();
    // If we're writing to this tensor, it's no longer constant
    if (is_write && value.isTensor()) {
      at::Tensor tensor = value.toTensor();
      // Undefined tensors are optional tensors.
      if (tensor.defined()) {
        auto *record = mapper.rawTensorRecord(tensor);
        if (record != nullptr) {
          record->is_const = false;
        }
      }
    }
    inputs.push_back(insertValueIntoGraph(value, graph, mapper));
  }
  return createAtenTarget(graph, schema, inputs, stack, mapper);
}

namespace {
// The in-place versions of these functions break the rule that in place ops
// don't change the shape of the tensor they operate on.
const std::unordered_set<std::string> in_place_reshapes{
    "aten::squeeze", "aten::unsqueeze", "aten::transpose"};
} // namespace

bool shouldRunOnCpu(bool is_inplace, const std::string &op_name) {
  return !is_inplace || in_place_reshapes.count(op_name) != 0;
}

HalfFloatConverter::HalfFloatConverter(c10::Stack *stack,
                                       const c10::FunctionSchema &schema,
                                       ValueMapper &mapper)
    : _stack(stack), _schema(schema), _mapper(mapper) {}

void HalfFloatConverter::pre() {
  // Convert any halves to floats. Keep track of what we replaced in case
  // the op reshapes any of them.
  _has_half_input = false;
  // The first pass looks for half-typed inputs, casts them to float, and places
  // them into the _half_map lookup table so the original half-typed tensors
  // can be appropriately reshaped after the op if necessary.
  // The second pass only runs if any half-typed input tensors were found in
  // the first pass. It tracks all of the float-typed output tensors as half
  // typed tensors (we make the assumption that if we supplied any half-typed
  // input tensors, any outputs will also be half-typed).
  for (size_t pass = 0; pass < 2; pass++) {
    if (pass == 1 && !_has_half_input) {
      break;
    }
    for (size_t i = 0; i < _stack->size(); i++) {
      auto &value = _stack->at(i);
      if (!value.isTensor()) {
        continue;
      }
      auto t = value.toTensor();
      if (!t.defined()) {
        continue;
      }
      auto tt = value.type()->cast<at::TensorType>();
      switch (pass) {
      case 0: {
        if (tt->scalarType() == at::ScalarType::Half) {
          at::Tensor t_cast = t.toType(at::ScalarType::Float);
          value = t_cast;
          _half_map[t_cast.getIntrusivePtr()] = t.getIntrusivePtr();
          _has_half_input = true;
        } else if (_mapper.isHalfTensor(t)) {
          _has_half_input = true;
        }
        break;
      }
      case 1: {
        const c10::Argument &argument = _schema.arguments()[i];
        if (tt->scalarType() == at::ScalarType::Float &&
            argument.alias_info() && argument.alias_info()->isWrite()) {
          _mapper.markHalfTensor(t);
        }
        break;
      }
      }
    }
  }
}

void HalfFloatConverter::post() {
  if (!_has_half_input) {
    return;
  }
  // First do any necessary reshapes on half-typed input tensors
  for (const auto &pair : _half_map) {
    at::Tensor float_tensor{pair.first};
    at::Tensor half_tensor{pair.second};
    if (half_tensor.sizes() != float_tensor.sizes()) {
      half_tensor.resize_as_(float_tensor);
    }
  }
  // Walk the stack noting any output tensors that should be considered
  // half-typed
  for (size_t i = 0; i < _stack->size(); i++) {
    auto &value = _stack->at(i);
    if (!value.isTensor()) {
      continue;
    }
    auto t = value.toTensor();
    if (!t.defined()) {
      continue;
    }
    auto tt = value.type()->cast<at::TensorType>();
    if (tt->scalarType() != at::ScalarType::Float) {
      continue;
    }
    _mapper.markHalfTensor(t);
  }
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
