// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "JitDispatch.hpp"

#include <string>
#include <utility>

#include "../../PoptorchSymbols.hpp"
#include "../../popart_canonicalization/PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Logging.hpp"

#include "../CommonHelperFunctions.hpp"

namespace poptorch {

namespace {

void fixFakeTargetOutput(torch::jit::Node *fake_target,
                         const c10::Stack &stack) {
  std::uint32_t index = 0;
  for (c10::IValue value : stack) {
    if (value.isTensor()) {
      torch::jit::Value *val = nullptr;

      // Add any missing outputs. They frequently return scalars which we just
      // ignore here as our canonicalisation only returns tensors.
      while (index >= fake_target->outputs().size()) {
        fake_target->addOutput();
      }
      at::Tensor tensor = value.toTensor();

      // Sometimes "Tensors" are actually "Not tensors" but still stored as a
      // tensor and will assert in the infer type.
      if (tensor.sizes().size() == 1 && tensor.sizes()[0] == 0) {
        continue;
      }
      val = fake_target->output(index);
      val->inferTypeFrom(tensor);
    }
    index++;
  }
}

} // namespace

void JITDispatch::createGraph(const std::vector<at::Tensor> &inputs,
                              const std::vector<at::Tensor> &parameters) {
  // We build up the torch IR graph as well.
  auto add = [&](at::Tensor &tensor) {
    torch::jit::Value *value = graph.addInput(tensor.name());
    value->inferTypeFrom(tensor);

    _mapper.addTensor(tensor, value);
  };

  // Add any inputs.
  for (at::Tensor tensor : inputs) {
    add(tensor);
  }

  // Add the parameters.
  for (at::Tensor tensor : parameters) {
    add(tensor);
  }
}

void JITDispatch::markOutputs(
    const std::vector<at::Tensor> &outputs,
    const std::vector<at::Tensor> &persistent_data_storage) {
  (void)persistent_data_storage;

  int64_t output_num = 0;
  for (const at::Tensor &tensor : outputs) {
    torch::jit::Value *val = _mapper.getValueForTensor(tensor);

    logging::trace(
        "[TRACING-2][JIT] Graph output: Tensor ptr {}, jit ir %{} {}",
        reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
        val->debugNameBase(), toString(tensor));

    graph.registerOutput(val);

    // For now, disable overlapping host IO on every output
    auto overlap_symbol = getOverlapSymbol("_for_output", output_num);
    const std::string value_str = "no_overlap";
    graph.return_node()->s_(overlap_symbol, value_str);

    output_num++;
  }
}

at::Tensor &JITDispatch::copyInplace(at::Tensor &self,
                                     const at::Tensor &other) {
  if (other.unsafeGetTensorImpl()->is_wrapped_number()) {
    torch::jit::Value *val = graph.insertConstant(other);
    _mapper.addTensor(other, val, true);
  }

  ValueMapper::TrackedTensor *dest = _mapper.rawTensorRecord(self);
  ValueMapper::TrackedTensor *src = _mapper.rawTensorRecord(other);
  ERROR_ON(dest == nullptr);
  ERROR_ON(src == nullptr);

  dest->jit = src->jit;
  dest->is_const = src->is_const;

  return self;
}

// _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device?
// device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat?
// memory_format=None) -> Tensor
// Apears in 1.10.
at::Tensor JITDispatch::toCopyInplace(
    const at::Tensor &self, c10::optional<at::ScalarType> /*dtype*/,
    c10::optional<at::Layout> /*layout*/, c10::optional<at::Device> /*device*/,
    c10::optional<bool> /*pin*/, c10::optional<c10::MemoryFormat> /*fmt*/) {
  // TODO(T45469): Renable.
  return self;
}

void JITDispatch::registerEmptyTensor(const at::Tensor &tensor) {
  torch::jit::Node *n =
      graph.createUninitialized(c10::TensorType::create(tensor));
  _mapper.addTensor(tensor, n->output(0), true);
}

at::Tensor JITDispatch::convolution(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, const at::IntArrayRef stride,
    const at::IntArrayRef padding, const at::IntArrayRef dilation,
    const bool transposed, const at::IntArrayRef output_padding,
    const int64_t groups) {

  c10::OperatorName name{"aten::convolution", ""};

  c10::Dispatcher &dispatcher = c10::Dispatcher::singleton();
  c10::OperatorHandle op = *dispatcher.findOp(name);
  const c10::FunctionSchema &schema = op.schema();

  // Turn our convolution inputs into a generic stack input.
  c10::Stack stack;
  stack.push_back(input);
  stack.push_back(weight);
  stack.push_back(bias);
  stack.push_back(stride);
  stack.push_back(padding);
  stack.push_back(dilation);
  stack.push_back(transposed);
  stack.push_back(output_padding);
  stack.push_back(groups);

  // Add it to the graph as a normal output.
  torch::jit::Node *fake_target =
      lowerFromSchema(schema, &stack, graph, _mapper);

  // Get the handler for the convolution.
  auto op_typed = op.typed<decltype(at::convolution)>();

  // Rerun on CPU to see the sizes.
  at::Tensor output = op_typed.redispatch(
      c10::DispatchKeySet({c10::DispatchKey::AutogradOther}), input, weight,
      bias, stride, padding, dilation, transposed, output_padding, groups);

  // Add the output into the stack.
  stack.clear();
  stack.push_back(output);

  // Fix the fake tensor so it can still work with our canonicalisation
  // functions which check the output.
  fixFakeTargetOutput(fake_target, stack);

  logging::trace("[TRACING-2][JIT] Node tensor output size: ={}",
                 output.sizes());

  // Run our normal canonicalisation passes on it.
  canonicaliseAndFixOutput(schema, stack, &fake_target);

  return output;
}

// aten::detach(Tensor(a) self) -> (Tensor(a))
at::Tensor JITDispatch::detach(const at::Tensor &self) { return self; }

// Convert the operation into our normal IR style operation.
void JITDispatch::canonicaliseAndFixOutput(const c10::FunctionSchema &schema,
                                           c10::Stack &stack,
                                           torch::jit::Node **fake_target) {
  torch::jit::Node *new_node = canonicalise(schema, *fake_target, graph, false);

  // Point fake_target at the new node
  *fake_target = new_node;

  logging::trace("[TRACING-2][JIT] Post canonicalisation {}", *new_node);

  // Fix up the outputs.
  std::uint32_t output_index = 0;
  for (c10::IValue value : stack) {
    // PopART doesn't always match these 1:1.
    if (output_index >= new_node->outputs().size()) {
      break;
    }

    // Start tracking the tensor.
    if (value.isTensor()) {
      at::Tensor tensor = value.toTensor();

      torch::jit::Value *val = new_node->output(output_index);
      // Check whether the handler replaced this value.
      auto *replacement = wasReplaced(val);
      if (replacement != nullptr) {
        val = replacement;
      }
      val->inferTypeFrom(tensor);
      _mapper.addTensor(tensor, val);

      logging::trace("[TRACING-2][JIT] Output: Tensor ptr {}, jit ir %{} {}",
                     reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
                     val->debugNameBase(), toString(tensor));

      output_index++;
    }
  }
}

void JITDispatch::fallback(const c10::OperatorHandle &initial_op,
                           c10::Stack *stack) {
  std::string name = initial_op.schema().operator_name().name;
  c10::Dispatcher &dispatcher = c10::Dispatcher::singleton();
  c10::OperatorHandle op = initial_op;

  // Run through the schema to find out if one of the operators is supposed to
  // be inplace.
  c10::intrusive_ptr<at::TensorImpl> inplace_tensor =
      getInplaceArgument(*stack, initial_op.schema());

  // If ends with '_', it's inplace. Remove the "_" and use the outplace version
  // instead.
  bool is_in_place = name[name.size() - 1] == '_';
  if (is_in_place) {
    // These are special cases because there is no zero / fill.
    if (name == "aten::zero_") {
      name = "aten::zeros_like";
    } else if (name == "aten::fill_") {
      name = "aten::full_like";
    } else {
      name.erase(name.end() - 1, name.end());
    }
    op = *dispatcher.findOp({name, ""});
  }

  const c10::FunctionSchema &schema = op.schema();

  // Create a fake IR node for us to target using the schema.
  torch::jit::Node *fake_target =
      lowerFromSchema(schema, stack, graph, _mapper);

  if (is_in_place) {
    // The Op is in place: we don't need to run the CPU version.
    // Just clear the stack and keep the first input.
    at::Tensor t = stack->at(0).toTensor();
    stack->clear();
    stack->push_back(t);
  } else {
    // Convert any halves to floats
    for (size_t i = 0; i < stack->size(); i++) {
      auto &value = stack->at(i);
      if (value.isTensor()) {
        auto tt = value.type()->cast<at::TensorType>();
        if (tt->scalarType() == at::ScalarType::Half) {
          at::Tensor t = value.toTensor();
          value = t.toType(at::ScalarType::Float);
        }
      }
    }
    // Call the CPU version to get the output shape
    dispatcher.callBoxed(op, stack);
  }

  // Fix the fake tensor so it can still work with our canonicalisation
  // functions which check the output.
  fixFakeTargetOutput(fake_target, *stack);
  logging::trace("[TRACING-2][JIT] Pre canonicalisation {}", *fake_target);

  // Run our normal canonicalisation passes on it.
  // The original fake_target node will be deleted but replaced with a new node.
  canonicaliseAndFixOutput(schema, *stack, &fake_target);

  logging::trace("[TRACING-2][JIT] Post canonicalisation and fix output {}",
                 *fake_target);

  std::size_t i = 0;
  for (c10::IValue value : *stack) {
    if (value.isTensor()) {
      at::Tensor tensor = value.toTensor();
      logging::trace(
          "[TRACING-2][JIT] Node tensor output at index {} size: ={}", i++,
          tensor.sizes());
    } else {
      logging::trace("[TRACING-2][JIT] Node scalar output at index {}", i++);
    }
  }

  // Switcheroo the output so the inplace tensor reference is now pointing to
  // the output.
  if (inplace_tensor) {
    at::Tensor inplace{inplace_tensor};
    at::Tensor output = stack->at(0).toTensor();

    // Get the jit value we are tracking for the output.
    torch::jit::Value *value = _mapper.getValueForTensor(output);

    // Overwrite the inplace tensor with that jit. Now a reference to the
    // inplace tensor correctly points to this outplace value.
    ValueMapper::TrackedTensor *record = _mapper.rawTensorRecord(inplace);
    ERROR_ON_MSG(
        !record,
        "[TRACING-2][JIT] Inplace op is not tracking inplace argument");
    record->jit = value;
  }

  logging::trace("[TRACING-2][JIT] Graph after interception of {}=\n{}\n",
                 schema.name(), graph);
}

} // namespace poptorch
