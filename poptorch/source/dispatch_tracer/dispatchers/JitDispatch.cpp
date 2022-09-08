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

#include "pytorch_bridge/CompilerOptions.hpp"

#include "../CommonHelperFunctions.hpp"
#include "../Tensor.hpp"

namespace poptorch {

class WithMetadata {
public:
  explicit WithMetadata(const std::string &metadata) {
    setCurrentMetadata(metadata);
  }
  ~WithMetadata() { setCurrentMetadata(""); }
};

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

JITDispatch::JITDispatch(const CompilerOptions &options)
    : graph(std::make_shared<torch::jit::Graph>()), _mlir_dispatch(options) {}

at::Tensor JITDispatch::allocateTensor(
    c10::IntArrayRef sizes, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Device> device, c10::optional<at::Layout> layout,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format) {
  // We delegate tensor creation to the MLIR dispatcher so we don't end up with
  // clashing IPU tensor IDs.
  return _mlir_dispatch.allocateTensor(sizes, dtype, device, layout, pin_memory,
                                       memory_format);
}

at::Tensor JITDispatch::addConstant(const at::Tensor &cpu_tensor) {
  ERROR_ON(!cpu_tensor.unsafeGetTensorImpl()->is_cpu());
  WithMetadata metadata("constant");

  auto src = copyAndCoerceType(cpu_tensor);

  at::Tensor tensor = allocateTensor(src.sizes(), src.scalar_type());

  auto *value = insertConstant(graph.get(), src);

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
  setSourceRangeToCurrentLocation(value->node());
  value->setType(c10::TensorType::create(tensor)->withRequiresGrad(
      cpu_tensor.requires_grad()));

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
  WithMetadata metadata("input");
  return addTensor(cpu_tensor, /* is_parameter= */ false);
}

at::Tensor JITDispatch::addParameter(const at::Tensor &tensor) {
  WithMetadata metadata("parameter");
  at::ScalarType type = tensor.scalar_type();
  // PopART doesn't allow non-floating point variables so add them as
  // constants instead. These will be deleted from parameters and buffers
  // in python before passed to lowering.
  if (!at::isFloatingType(type)) {
    return addConstant(tensor);
  }
  return addTensor(tensor, /* is_parameter= */ true);
}

void JITDispatch::addOutput(const at::Tensor &ipu_src,
                            const at::Tensor &cpu_dest) {
  WithMetadata metadata("output");
  // The PopART backend will allocate its own buffers: ignore cpu_dest.
  UNUSED(cpu_dest);
  auto *record = _mapper.rawTensorRecord(ipu_src);
  ERROR_ON_MSG(record == nullptr,
               "Internal: graph output tensor not present in value mapper "
                   << static_cast<void *>(&_mapper) << " for "
                   << static_cast<void *>(ipu_src.unsafeGetTensorImpl()));

  torch::jit::Value *val = record->jit;

  // If the output is an input: add an identity op to make sure the graph
  // is not empty.
  // TODO(T62169) handle empty graphs better.
  for (auto *i : graph->inputs()) {
    if (i == val) {
      auto *none = graph->createNone();
      insertNodeInGraph(graph.get(), none);
      val = createAndInsertNode(graph.get(), c10::aten::clone,
                                {val, none->output()}, ImplicitCast::None,
                                OutputType::AsFirstInput)
                ->output();
      break;
    }
  }

  logging::trace("[TRACING-2][JIT] Graph output: Tensor ptr {}, jit ir %{} "
                 "(scalar type {})",
                 reinterpret_cast<void *>(ipu_src.unsafeGetTensorImpl()),
                 val->debugNameBase(),
                 val->type()->expect<c10::TensorType>()->scalarType().value_or(
                     at::ScalarType::Undefined));

  graph->registerOutput(val);
}

void JITDispatch::finalizeGraph() {
  logging::trace("[TRACING-2][JIT] Graph after marking outputs\n{}\n", *graph);
  // Clear the code location
  setCurrentPythonCodeLocation({});
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

  WithMetadata meta("copy_");
  torch::jit::Value *copy =
      createAndInsertNode(graph.get(), c10::aten::copy_,
                          {self_tracked->jit, src_tracked->jit},
                          ImplicitCast::None, OutputType::AsFirstInput, 1)
          ->output();

  copy->setType(
      self_tracked->jit->type()->expect<c10::TensorType>()->withRequiresGrad(
          src_tracked->jit->type()->expect<c10::TensorType>()->requiresGrad()));

  // The copy_ node's output is now an alias of self, so mark it as such; this
  // essentially mirrors what JITDispatch::fallback would do.
  auto *aliased_input = _inplace_tracker.eraseCurrentAlias(self_tracked->jit);
  if (aliased_input != nullptr) {
    _inplace_tracker.registerAlias(aliased_input, copy);
  }

  self_tracked->jit = copy;

  self.set_requires_grad(src.requires_grad());
  logging::trace("[TRACING-2][JIT] copyInplace: self tensor new jit ir %{}",
                 self_tracked->jit->debugName());

  return self;
}

void JITDispatch::registerEmptyTensor(const at::Tensor &tensor) {
  WithMetadata metadata("empty");
  // Do not call copyAndCoerceType from this method:
  // the source tensor hasn't been added to the mapper yet.

  // The tensor shouldn't need converting anyway: it should be created with a
  // valid type.
  auto coerced_scalar_type = coerceToSupportedType(tensor.scalar_type());
  ERROR_ON_MSG(
      coerced_scalar_type != tensor.scalar_type(),
      "[Internal error] The empty tensor should have a valid compiler type");
  // aten::empty.memory_format(int[] size, *, ScalarType? dtype=None,
  //                           Layout? layout=None, Device? device=None,
  //                           bool? pin_memory=None,
  //                           MemoryFormat? memory_format=None) -> Tensor
  auto *g = graph.get();
  auto *pin_memory = g->createNone();
  auto *memory_format = g->createNone();
  insertNodeInGraph(g, pin_memory);
  insertNodeInGraph(g, memory_format);
  torch::jit::Node *n = createAndInsertNode(
      g, c10::aten::empty,
      {insertConstant(g, tensor.sizes()),
       insertConstant(g, tensor.scalar_type()),
       insertConstant(g, tensor.layout()), insertConstant(g, tensor.device()),
       pin_memory->output(), memory_format->output()});
  n->output()->inferTypeFrom(tensor);
  setSourceRangeToCurrentLocation(n);
  _mapper.addTensor(tensor, n->output());
}

// aten::detach(Tensor(a) self) -> (Tensor(a))
void JITDispatch::detach(const c10::OperatorHandle &op, c10::Stack *stack,
                         bool moving_parameters) {
  // We only handle the special case when we're moving parameters here. If we're
  // not moving parameters, we'll defer to the fallback and actually create a
  // dispatch op on the PopART graph.
  if (!moving_parameters) {
    fallback(op, stack);
    return;
  }

  const c10::FunctionSchema &schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  const auto arguments = torch::jit::last(stack, num_arguments);

  ERROR_ON(arguments.size() != 1);
  at::Tensor in = arguments.front().toTensor();

  at::Tensor out(in.unsafeGetTensorImpl()->shallow_copy_and_detach(
      /*version_counter=*/in.unsafeGetTensorImpl()->version_counter(),
      /*allow_tensor_metadata_change=*/true));

  // The new tensor points at the same mlir tensor as the source.
  _mapper.addTensor(out, _mapper.getValueForTensor(in));

  torch::jit::drop(stack, num_arguments);
  torch::jit::push(stack, out);
}

const std::vector<std::vector<char>> &
JITDispatch::getSourceLocationExcludes() const {
  return _mlir_dispatch.getSourceLocationExcludes();
}
void JITDispatch::setCurrentCodeLocation(
    const torch::jit::SourceRange &source_location) {
  setCurrentPythonCodeLocation(source_location);
}

// Convert the operation into our normal IR style operation.
void JITDispatch::fixOutput(c10::Stack &stack, torch::jit::Node *node) {
  // Fix up the outputs.
  std::uint32_t output_index = 0;
  for (c10::IValue value : stack) {
    // Add any missing outputs. They frequently return scalars which we just
    // ignore here as our canonicalisation only returns tensors.
    while (output_index >= node->outputs().size()) {
      node->addOutput();
    }

    // Start tracking the output tensors, i.e. add them to the value mapper.
    torch::jit::Value *val = node->output(output_index);

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
      insertNodeInGraph(graph.get(), unpack);

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

void JITDispatch::fallback(const c10::OperatorHandle &op, c10::Stack *stack) {
  const c10::FunctionSchema &schema = op.schema();
  // Run through the schema to find out if one of the operators is supposed to
  // be inplace, this could be the 'out' argument of a non-inplace op.
  std::vector<at::Tensor> inplace_tensors = getInplaceArguments(*stack, schema);
  torch::jit::Value *aliased_input = nullptr;
  if (!inplace_tensors.empty()) {
    aliased_input = _inplace_tracker.eraseCurrentAlias(
        _mapper.getValueForTensor(inplace_tensors[0]));
  }

  // Tag all the nodes created by the handler with the initial schema string
  // representation so that they can be traced back to top level ops in the
  // profiler.
  WithMetadata metadata(c10::toString(schema));

  // Create a fake IR node for us to target using the schema.
  torch::jit::Node *node = lowerFromSchema(schema, stack, *graph, _mapper);
  logging::trace("[TRACING-2][JIT] Node from schema {}", *node);

  if (!inplace_tensors.empty()) {
    // For inplace ops, cast all input tensors to the same type as the output
    // tensor.
    auto output_type = inplace_tensors[0].scalar_type();
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
      auto *g = graph.get();
      auto *dtype = insertConstant(g, output_type);
      auto *non_blocking = insertConstant(g, false);
      auto *copy = insertConstant(g, false);
      auto *none = g->createNone();
      insertNodeInGraph(g, none);
      auto *cast =
          createAndInsertNode(graph.get(), c10::aten::to,
                              {jv, dtype, non_blocking, copy, none->output()});
      cast->output()->setType(
          jv->type()->expect<c10::TensorType>()->withScalarType(output_type));
      // The cast needs to be before the node.
      cast->moveBefore(node);
      dtype->node()->moveBefore(cast);
      non_blocking->node()->moveBefore(cast);
      copy->node()->moveBefore(cast);
      none->moveBefore(cast);

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
  fixOutput(*stack, node);

  logging::trace("[TRACING-2][JIT] Pre canonicalisation {}", *node);

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
  if (!inplace_tensors.empty()) {
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
        _mapper.rawTensorRecord(inplace_tensors[0]);
    ERROR_ON_MSG(
        !record,
        "[TRACING-2][JIT] Inplace op is not tracking inplace argument");

    // Ensure the value and torch tensor shapes match
    JitTensorInfo value_info(value);
    inplace_tensors[0].unsafeGetTensorImpl()->set_sizes_contiguous(
        value_info.dims);

    // Validate to make sure the data type also matches.
    validateTensorShapeAndType(value, inplace_tensors[0]);
    record->jit = value;
  }

  if (logging::shouldLog(logging::Level::Trace)) {
    logging::trace("[TRACING-2][JIT] Graph after interception of {}=\n{}\n",
                   schema, truncateGraphString(*graph));
  }
}

InplaceGraphInfo
JITDispatch::finalizeInplaceGraphInfo(size_t num_anchors,
                                      bool replicas_needing_broadcast) {
  return _inplace_tracker.finalizeGraph(*graph, num_anchors,
                                        replicas_needing_broadcast);
}

} // namespace poptorch
