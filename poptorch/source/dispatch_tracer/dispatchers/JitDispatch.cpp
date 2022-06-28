// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "JitDispatch.hpp"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

#include "../../PoptorchSymbols.hpp"
#include "../../popart_canonicalization/PopartCanonicalizationUtils.hpp"

#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "../CommonHelperFunctions.hpp"
#include "../Tensor.hpp"

namespace poptorch {

std::string truncateGraphString(torch::jit::Graph &graph) {
  static const int num_lines_max = [=]() {
    if (const char *graph_len = std::getenv("POPTORCH_MAX_GRAPH_LEN")) {
      const int n = std::stoi(graph_len);
      logging::trace("POPTORCH_MAX_GRAPH_LEN={}", n);
      return n;
    }
    const int n = 10;
    logging::trace("POPTORCH_MAX_GRAPH_LEN not set, defaulting to {}", n);
    return n;
  }();

  std::string s = graph.toString();
  if (num_lines_max <= 0 || s.empty()) {
    return s;
  }
  size_t start = s.size();
  for (int i = 0; i < num_lines_max; i++) {
    start = s.rfind('\n', start - 1);
    if (start == std::string::npos) {
      // Didn't find another new line: print everything.
      return s;
    }
  }
  // Start after the last line return.
  return "[...truncated...]" + s.substr(start);
}

at::Tensor JITDispatch::addConstant(const at::Tensor &cpu_tensor) {
  ERROR_ON(!cpu_tensor.unsafeGetTensorImpl()->is_cpu());

  auto src = copyAndCoerceType(cpu_tensor);

  at::Tensor tensor =
      allocateTensor(src.sizes(), c10::typeMetaToScalarType(src.dtype()));

  torch::jit::Value *value = makeConstant(*graph, src);

  logging::trace("[TRACING-2] Adding constant: Value {} with cpu ptr {}",
                 static_cast<void *>(value), cpu_tensor.data_ptr());

  _mapper.addTensor(tensor, value);
  setIsParameter(tensor, false);
  return tensor;
}

void errorOnZeroSizedTensor(const at::Tensor &tensor) {
  auto sizes = tensor.sizes();
  if (std::any_of(sizes.begin(), sizes.end(),
                  [](auto dim) { return dim == 0; })) {
    std::stringstream err;
    err << "Zero-sized tensors are unsupported (Got shape [";
    for (std::size_t i = 0; i < sizes.size() - 1; i++) {
      err << sizes[i] << ", ";
    }
    err << sizes[sizes.size() - 1] << "]).";
    ERROR(err.str());
  }
}

at::Tensor JITDispatch::addTensor(const at::Tensor &cpu_tensor,
                                  bool is_parameter) {
  ERROR_ON(!cpu_tensor.unsafeGetTensorImpl()->is_cpu());
  errorOnZeroSizedTensor(cpu_tensor);

  at::Tensor tensor = allocateTensor(
      cpu_tensor.sizes(), c10::typeMetaToScalarType(cpu_tensor.dtype()));

  tensor = copyAndCoerceType(tensor);

  torch::jit::Value *value = graph->addInput(cpu_tensor.name());
  value->inferTypeFrom(tensor);

  logging::trace("[TRACING-2] Adding {}: Value {} with cpu ptr {}",
                 is_parameter ? "parameter" : "input",
                 static_cast<void *>(value), cpu_tensor.data_ptr());
  copyDataFromCpuSource(tensor, cpu_tensor);
  _inplace_tracker.addTensor(value);
  _mapper.addTensor(tensor, value);
  setIsParameter(tensor, is_parameter);
  return tensor;
}

at::Tensor JITDispatch::addInput(const at::Tensor &cpu_tensor) {
  return addTensor(cpu_tensor, /* is_parameter= */ false);
}

at::Tensor JITDispatch::addParameter(const at::Tensor &tensor) {
  at::ScalarType type = tensor.scalar_type();
  // PopART doesn't allow non-floating point variables so add them as
  // constants instead. These will be deleted from parameters and buffers
  // in python before passed to lowering.
  if (!at::isFloatingType(type)) {
    return addConstant(tensor);
  }
  return addTensor(tensor, /* is_parameter= */ true);
}

void JITDispatch::createGraph() {
  graph = std::make_shared<torch::jit::Graph>();

  // No need to create a MLIR graph, we're going to only use the dispatcher
  // for shape inference, so just initialise the compiler.
  _mlir_dispatch.initCompiler();
}

void JITDispatch::addOutput(const at::Tensor &ipu_src,
                            const at::Tensor &cpu_dest) {
  // The PopART backend will allocate its own buffers: ignore cpu_dest.
  UNUSED(cpu_dest);
  auto *record = _mapper.rawTensorRecord(ipu_src);
  ERROR_ON_MSG(record == nullptr,
               "Internal: graph output tensor not present in the value mapper");

  torch::jit::Value *val;
  if (record->is_empty) {
    val = makeConstant(*graph, ipu_src);
    _mapper.addTensor(ipu_src, val);
  } else {
    val = record->jit;

    // If the output is an input: add an identity op to make sure the graph
    // is not empty.
    // TODO(T62169) handle empty graphs better.
    for (auto *i : graph->inputs()) {
      if (i == val) {
        val = createIdentity(graph.get(), {val})->output();
        break;
      }
    }
  }

  logging::trace("[TRACING-2][JIT] Graph output: Tensor ptr {}, jit ir %{} "
                 "(scalar type {})",
                 reinterpret_cast<void *>(ipu_src.unsafeGetTensorImpl()),
                 val->debugNameBase(),
                 val->type()->expect<c10::TensorType>()->scalarType().value_or(
                     at::ScalarType::Undefined));

  graph->registerOutput(val);
  // For now, disable overlapping host IO on every output
  auto overlap_symbol = getOverlapSymbol("_for_output", _next_output_idx);
  const std::string value_str = "no_overlap";
  graph->return_node()->s_(overlap_symbol, value_str);

  _next_output_idx++;
}

void JITDispatch::finalizeGraph() {
  logging::trace("[TRACING-2][JIT] Graph after marking outputs\n{}\n", *graph);
}

// copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
const at::Tensor &JITDispatch::copyInplace(const at::Tensor &self,
                                           const at::Tensor &src) {
  ValueMapper::TrackedTensor *self_tracked = _mapper.rawTensorRecord(self);
  ValueMapper::TrackedTensor *src_tracked = _mapper.rawTensorRecord(src);
  ERROR_ON(self_tracked == nullptr);
  ERROR_ON(src_tracked == nullptr);

  logging::trace(
      "[TRACING-2][JIT] copyInplace: src tensor {} (jit ir %{}), self tensor "
      "{} (jit ir %{})",
      static_cast<void *>(src.unsafeGetTensorImpl()),
      src_tracked->jit->debugName(),
      static_cast<void *>(self.unsafeGetTensorImpl()),
      self_tracked->jit->debugName());

  auto self_st =
      self_tracked->jit->type()->expect<c10::TensorType>()->scalarType();
  auto src_st =
      src_tracked->jit->type()->expect<c10::TensorType>()->scalarType();
  ERROR_ON(!self_st.has_value());
  ERROR_ON(!src_st.has_value());

  torch::jit::Value *copy;
  if (*self_st == *src_st) {
    copy = createIdentity(graph.get(), {src_tracked->jit})->output();
    copy->setType(self_tracked->jit->type());
  } else {
    copy = createCast(graph.get(), src_tracked->jit, *self_st)->output();
  }

  self_tracked->jit = copy;
  self_tracked->is_empty = src_tracked->is_empty;

  logging::trace("[TRACING-2][JIT] copyInplace: self tensor new jit ir %{}",
                 self_tracked->jit->debugName());

  return self;
}

void JITDispatch::registerEmptyTensor(const at::Tensor &tensor) {
  // Do not call copyAndCoerceType from this method:
  // the source tensor hasn't been added to the mapper yet.

  // The tensor shouldn't need converting anyway: it should be created with a
  // valid type.
  auto coerced_scalar_type = coerceToSupportedType(tensor.scalar_type());
  ERROR_ON_MSG(
      coerced_scalar_type != tensor.scalar_type(),
      "[Internal error] The empty tensor should have a valid compiler type");
  torch::jit::Node *n =
      graph->createUninitialized(c10::TensorType::create(tensor));
  _mapper.addTensor(tensor, n->output(0), true);
}

// aten::detach(Tensor(a) self) -> (Tensor(a))
at::Tensor JITDispatch::detach(const at::Tensor &self) {
  at::Tensor out(self.unsafeGetTensorImpl()->shallow_copy_and_detach(
      /*version_counter=*/self.unsafeGetTensorImpl()->version_counter(),
      /*allow_tensor_metadata_change=*/true));

  // The new tensor points at the same mlir tensor as the source.
  _mapper.addTensor(out, _mapper.getValueForTensor(self));
  return out;
}

void JITDispatch::setCurrentCodeLocation(
    const torch::jit::SourceRange &source_location) {
  setCurrentPythonCodeLocation(source_location);
}

// Convert the operation into our normal IR style operation.
void JITDispatch::canonicaliseAndFixOutput(const c10::FunctionSchema &schema,
                                           c10::Stack &stack,
                                           torch::jit::Node **node) {
  torch::jit::Node *new_node =
      canonicalise(schema, *node, *graph, /* is_allowed_to_fail=*/false);
  *node = new_node;

  logging::trace("[TRACING-2][JIT] Post canonicalisation {}", *new_node);

  // Fix up the outputs.
  std::uint32_t output_index = 0;
  for (c10::IValue value : stack) {
    // PopART doesn't always match these 1:1.
    if (output_index >= new_node->outputs().size()) {
      break;
    }

    // Start tracking the output tensors, i.e. add them to the value mapper.
    torch::jit::Value *val = new_node->output(output_index);
    // Check whether the handler replaced this value.
    auto *replacement = wasReplaced(val);
    if (replacement != nullptr) {
      val = replacement;
    }

    if (value.isTensor()) {
      at::Tensor tensor = value.toTensor();

      val->inferTypeFrom(copyAndCoerceType(tensor));
      _mapper.addTensor(tensor, val);

      logging::trace(
          "[TRACING-2][JIT] Output: Tensor ptr {}, jit ir %{} (scalar type {})",
          reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
          val->debugNameBase(),
          val->type()->expect<c10::TensorType>()->scalarType().value_or(
              at::ScalarType::Undefined));
    } else if (value.isTensorList()) {
      logging::trace("[TRACING-2][JIT] Output tensor list: jit ir %{}",
                     val->debugName());
      val->setType(value.type()->expect<c10::ListType>());
      auto tensor_list = value.toTensorVector();
      // Always insert list unpack if output value is a list.
      auto *unpack = graph->createListUnpack(val, tensor_list.size());
      graph->insertNode(unpack);

      for (size_t i = 0; i < tensor_list.size(); ++i) {
        at::Tensor tensor = tensor_list.at(i);
        val = unpack->output(i);
        val->inferTypeFrom(copyAndCoerceType(tensor));
        _mapper.addTensor(tensor, val);
        logging::trace("[TRACING-2][JIT] Output tensor list element: Tensor "
                       "ptr {}, jit ir %{} {}",
                       reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
                       val->debugNameBase(), toString(tensor));
      }
    }

    output_index++;
  }
}

void JITDispatch::fallback(const c10::OperatorHandle &initial_op,
                           c10::Stack *stack) {
  c10::Dispatcher &dispatcher = c10::Dispatcher::singleton();

  const c10::FunctionSchema &initial_schema = initial_op.schema();
  // Run through the schema to find out if one of the operators is supposed to
  // be inplace, this could be the 'out' argument of a non-inplace op.
  std::optional<at::Tensor> inplace_tensor =
      getInplaceArgument(*stack, initial_schema);
  torch::jit::Value *aliased_input = nullptr;
  if (inplace_tensor) {
    aliased_input = _inplace_tracker.eraseCurrentAlias(
        _mapper.getValueForTensor(*inplace_tensor));
  }

  c10::OperatorHandle op = getOutplaceOpHandle(initial_op, dispatcher);
  const c10::FunctionSchema &schema = op.schema();

  // Create a fake IR node for us to target using the schema.
  torch::jit::Node *node = lowerFromSchema(schema, stack, *graph, _mapper);

  if (inplace_tensor) {
    // For inplace ops, cast all input tensors to the same type as the output
    // tensor.
    auto output_type = inplace_tensor->scalar_type();
    bool output_float = c10::isFloatingType(output_type);
    for (size_t i = 0; i < stack->size(); i++) {
      const c10::IValue &sv = (*stack)[i];
      if (!sv.isTensor()) {
        continue;
      }
      const at::Tensor &tensor = sv.toTensor();
      auto input_type = tensor.scalar_type();
      bool input_float = c10::isFloatingType(input_type);
      if (input_type == at::ScalarType::Undefined ||
          input_type == output_type || input_float != output_float ||
          !canCast(input_type, output_type)) {
        continue;
      }
      torch::jit::Value *jv = node->input(i);
      auto *cast = createCast(graph.get(), jv, output_type);
      node->replaceInputWith(jv, cast->output());
    }
  }

  // The MLIR dispatcher is going to use the shape and type of the inputs to
  // infer the shape and type of the outputs so we need to create dummy MLIR
  // tensors for each input.
  std::function<void(const c10::IValue &value)> process_value =
      [&](const c10::IValue &value) {
        if (value.isList()) {
          for (const auto &v : value.toList()) {
            process_value(v);
          }
        } else if (value.isTensor()) {
          const at::Tensor &tensor = value.toTensor();
          // Sometimes Undefined is used to mark an optional tensor as not set.
          if (tensor.scalar_type() == at::ScalarType::Undefined) {
            ERROR_ON_MSG(
                tensor.numel() != 0,
                "[Internal error] Non-empty tensor of type 'Undefined'");
            // No need to register the tensor if it's undefined.
            return;
          }
          // If the tensor is not tracked by JIT then don't track it in MLIR.
          // (It's probably a CPU constant)
          if (_mapper.getValueForTensor(tensor) != nullptr) {
            _mlir_dispatch.registerEmptyTensor(tensor);
          }
        } else {
          // If this assertion is hit then we need to add support for this kind
          // of value by going through the container and identifying all the
          // tensors.
          ERROR_ON_MSG(value.isTuple() || value.isGenericDict(),
                       "[Internal] Support for container "
                           << value.tagKind() << " not implemented");
        }
      };
  for (const c10::IValue &value : *stack) {
    process_value(value);
  }
  _mlir_dispatch.handleOp(op, stack);
  // Fix the fake tensor so it can still work with our canonicalisation
  // functions which check the output.
  fixNodeOutput(node, *stack);
  logging::trace("[TRACING-2][JIT] Pre canonicalisation {}", *node);

  // Run our normal canonicalisation passes on it.
  // The original node will be deleted but replaced with a new node.
  canonicaliseAndFixOutput(schema, *stack, &node);

  // Annotate for loops as subgraphs.
  annotateSubgraphsDispatch(graph.get(), node);

  logging::trace("[TRACING-2][JIT] Post canonicalisation and fix output {}",
                 *node);

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
    at::Tensor output = stack->at(0).toTensor();

    // Get the jit value we are tracking for the output.
    torch::jit::Value *value = _mapper.getValueForTensor(output);
    // If the modified inplace tensor was an alias for an input then
    // register the new alias.
    if (aliased_input != nullptr) {
      _inplace_tracker.registerAlias(aliased_input, value);
    }

    // Overwrite the inplace tensor with that jit. Now a reference to the
    // inplace tensor correctly points to this outplace value.
    ValueMapper::TrackedTensor *record =
        _mapper.rawTensorRecord(*inplace_tensor);
    ERROR_ON_MSG(
        !record,
        "[TRACING-2][JIT] Inplace op is not tracking inplace argument");
    record->jit = value;
    record->is_empty = false;
  }

  if (logging::shouldLog(logging::Level::Trace)) {
    logging::trace("[TRACING-2][JIT] Graph after interception of {}=\n{}\n",
                   schema.name(), truncateGraphString(*graph));
  }
}

InplaceGraphInfo
JITDispatch::finalizeInplaceGraphInfo(size_t num_anchors,
                                      bool replicas_needing_broadcast) {
  return _inplace_tracker.finalizeGraph(*graph, num_anchors,
                                        replicas_needing_broadcast);
}

} // namespace poptorch
