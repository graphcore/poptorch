// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "MlirDispatch.hpp"

#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "../../PoptorchSymbols.hpp"
#include "../../popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "MlirDispatchUtils.hpp"
#include "poptorch/DispatchTracer.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Logging.hpp"

#include "../CommonHelperFunctions.hpp"

namespace poptorch {

MLIRExecutable::MLIRExecutable(
    std::unique_ptr<poptorch_ir::PoptorchExecutorWrapper> &&other)
    : _impl(std::move(other)) {}

MLIRExecutable::~MLIRExecutable() {}

void MLIRExecutable::execute(const std::vector<at::Tensor> &inputs) {
  std::vector<void *> ptrs;
  ptrs.resize(inputs.size());

  // Keep the refs around.
  std::vector<at::Tensor> converted;

  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const at::Tensor &tensor = inputs[i];
    if (tensor.scalar_type() == at::ScalarType::Long) {
      converted.push_back(tensor.to(at::ScalarType::Int));
      ptrs[i] = converted.back().data_ptr();
    } else {
      ptrs[i] = tensor.data_ptr();
    }
  }

  _impl->execute(ptrs);
  _impl->weightsToHost();
}

void MLIRExecutable::weightsToDevice() { _impl->weightsToDevice(); }

void MLIRExecutable::weightsToHost() { _impl->weightsToHost(); }

MLIRDispatch::MLIRDispatch() { this->generateDispatchTable(); }

static poptorch_ir::Type toCompilerType(const at::ScalarType &elem_type) {
  switch (elem_type) {
  case at::ScalarType::Bool:
    return poptorch_ir::Type::BOOL;
  case at::ScalarType::Byte:
    return poptorch_ir::Type::UNSIGNED_CHAR;
  case at::ScalarType::Char:
    return poptorch_ir::Type::CHAR;
  case at::ScalarType::Float:
    return poptorch_ir::Type::FLOAT;
  case at::ScalarType::Half:
    return poptorch_ir::Type::HALF;
  case at::ScalarType::Short:
    return poptorch_ir::Type::SHORT;
  case at::ScalarType::Int:
  case at::ScalarType::Long: // We will convert this.
    return poptorch_ir::Type::INT;
  default:
    ERROR("Unsupported tensor input type from pytorch.");
  }
}

static poptorch_ir::Type toCompilerType(const at::Tensor &tensor) {
  at::ScalarType elem_type = tensor.scalar_type();

  return toCompilerType(elem_type);
}

void MLIRDispatch::createGraph(const std::vector<at::Tensor> &inputs,
                               const std::vector<at::Tensor> &parameters) {
  // Init our MLIR compiler.
  _compiler.init();

  // Start timing how long it takes us to build the graph.
  _compiler.startTraceTiming();

  _last_processed_node = nullptr;

  // We build up the torch IR graph as well.
  const auto add_to_jit = [&](const at::Tensor &tensor) {
    torch::jit::Value *value = _graph.addInput(tensor.name());
    value->inferTypeFrom(tensor);

    _mapper.addTensor(tensor, value);
  };

  int i = 0;
  // Add any inputs.
  for (const at::Tensor &tensor : inputs) {
    add_to_jit(tensor);

    const std::string str = "Input/" + std::to_string(i++);
    std::vector<std::int64_t> shape;
    shape.reserve(tensor.dim());
    for (std::int64_t dim : tensor.sizes()) {
      shape.push_back(dim);
    }

    logging::trace("[TRACING-2] Adding input: {} ", tensor.data_ptr());
    poptorch_ir::TensorId value = _compiler.addInput(
        tensor.data_ptr(), shape, toCompilerType(tensor), str.c_str());
    _mapper.addTensor(tensor, value);
  }

  i = 0;
  // Add the parameters.
  for (const at::Tensor &tensor : parameters) {
    add_to_jit(tensor);

    const std::string str = "Parameter/" + std::to_string(i++);

    std::vector<std::int64_t> shape;
    shape.reserve(tensor.dim());
    for (std::int64_t dim : tensor.sizes()) {
      shape.push_back(dim);
    }

    logging::trace(
        "[TRACING-2] Adding parameter: {} with storage {}", tensor.data_ptr(),
        static_cast<void *>(tensor.storage().unsafeGetStorageImpl()));

    poptorch_ir::TensorId value = _compiler.addParameter(
        tensor.data_ptr(), shape, toCompilerType(tensor), str.c_str());
    _mapper.addTensor(tensor, value);
  }
}

void MLIRDispatch::markOutputs(
    const std::vector<at::Tensor> &ids,
    const std::vector<at::Tensor> &persistent_data_storage,
    const std::string &output_structure) {
  UNUSED(output_structure);
  ERROR_ON_MSG(
      ids.size() != persistent_data_storage.size(),
      "[INTERNAL] Outputs and output storages do not have same length");

  for (std::size_t i = 0; i < ids.size(); ++i) {
    at::Tensor tensor = ids[i];
    void *storage = persistent_data_storage[i].data_ptr();

    const std::string str = "Output/" + std::to_string(i);
    poptorch_ir::TensorId id = _mapper.getMLIRForTensor(tensor);
    logging::trace(
        "[TRACING-2] Marking output: {} with storage {}",
        static_cast<void *>(tensor.unsafeGetTensorImpl()),
        static_cast<void *>(tensor.storage().unsafeGetStorageImpl()));

    if (id != poptorch_ir::tensor_error_id && id != poptorch_ir::none_id) {
      _compiler.addOutput(id, storage, str.c_str());
    }
  }

  _compiler.addReturn();
  _compiler.endTraceTiming();
}

std::vector<poptorch_ir::TensorId>
MLIRDispatch::mlirFromStack(c10::Stack &stack) {
  std::vector<poptorch_ir::TensorId> ids;

  // For each IValue (which may or may not be a tensor).
  for (c10::IValue value : stack) {
    // Look up the MLIR value if it is a tensor. Other stuff will be in the JIT
    // IR. (This is only used for JIT.)
    if (value.isTensor()) {
      at::Tensor tensor = value.toTensor();
      ids.push_back(findTensor(tensor));
    }
  }

  return ids;
}

const at::Tensor &MLIRDispatch::copyInplace(const at::Tensor &self,
                                            const at::Tensor &src) {
  // We have to look the tensors up first as they might be a special case.
  _compiler.copy_(findTensor(self), findTensor(src));

  // Convert the tensors to MLIR and create the copy.
  ValueMapper::TrackedTensor *self_info = _mapper.rawTensorRecord(self);
  ValueMapper::TrackedTensor *src_info = _mapper.rawTensorRecord(src);

  // Update the tensor info so it now refers to the other metadata.
  self_info->jit = src_info->jit;
  self_info->mlir = src_info->mlir;
  return self;
}

void packStack(c10::Stack & /*unused*/) {}

// A small helper to populate the c10::stack.
template <typename T, typename... Args>
void packStack(c10::Stack &stack, T &arg, Args... args) {
  stack.push_back(arg);
  packStack(stack, args...);
}

at::Tensor MLIRDispatch::convolution(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, const at::IntArrayRef strides,
    const at::IntArrayRef padding, const at::IntArrayRef dilation,
    const bool transposed, const at::IntArrayRef output_padding,
    const int64_t groups) {
  // Create the stack which is just a vector of IValues, I.e all of the above
  // arguments.
  c10::Stack stack;

  // An optional bias is actually an undefined tensor, not 100% why they have
  // two layers of indirection (optional AND undefined).
  packStack(stack, input, weight, *bias, strides, padding, dilation, transposed,
            output_padding, groups);

  // Unpack the above and create the MLIR convolution node.
  this->convolution(stack);

  // Add a JIT node to keep the jit graph clean-ish, we don't bother adding the
  // inputs.
  torch::jit::Node *node = _graph.create(c10::aten::convolution, {}, 1);
  _graph.appendNode(node);

  // Map the input tensor onto the JIT.
  at::Tensor new_output = stack.at(0).toTensor();
  node->output(0)->inferTypeFrom(new_output);
  _mapper.addTensor(new_output, node->output(0));

  // Update the 'tail' so if another node needs to process the JIT it will
  // start from here.
  _last_processed_node = node;

  return new_output;
}

// _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device?
// device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat?
// memory_format=None) -> Tensor
// Apears in 1.10.
at::Tensor MLIRDispatch::toCopyInplace(const at::Tensor &self,
                                       c10::optional<at::ScalarType> dtype,
                                       c10::optional<at::Layout> layout,
                                       c10::optional<at::Device> device,
                                       c10::optional<bool> pin,
                                       c10::optional<c10::MemoryFormat> fmt) {
  // Create the new output tensor.
  at::Tensor out =
      at::native::empty_cpu(self.sizes(), dtype, layout, device, pin, fmt);

  // Zero tensor as it's possible that the tensor is accessable by the user
  // after tracing.
  at::zero_(out);

  // Make an empty IR node.
  // (empty_tensor returns an ODSTensorResults instance, which is a vector of a
  // vector of a tensor)
  poptorch_ir::TensorId out_id =
      _compiler.empty_tensor(self.sizes().vec(), toCompilerType(*dtype))
          .at(0)
          .at(0);

  // Track it in our mapper.
  _mapper.addTensor(out, out_id);

  // Create a normal copy in the graph (this does not alter the at::tensor)
  return copyInplace(out, self);
}

void MLIRDispatch::registerEmptyTensor(const at::Tensor &tensor) {
  // Don't bother intercepting on JIT as we automatically promote unknown nodes
  // to empty tensors.
  // (empty_tensor returns an ODSTensorResults instance, which is a vector of a
  // vector of the tensor)
  poptorch_ir::TensorId id =
      _compiler.empty_tensor(tensor.sizes().vec(), toCompilerType(tensor))
          .at(0)
          .at(0);
  _mapper.addTensor(tensor, id);
}

// aten::detach(Tensor(a) self) -> (Tensor(a))
at::Tensor MLIRDispatch::detach(const at::Tensor &self) { return self; }

void MLIRDispatch::fallback(const c10::OperatorHandle &op, c10::Stack *stack) {
  const c10::FunctionSchema &schema = op.schema();
  torch::jit::Symbol symbol = torch::jit::Symbol::fromQualString(schema.name());

  // Unfortunately we can't overload based only on the schema symbol as it does
  // not contain the overload info.
  std::string schema_key;
  if (schema.overload_name().empty()) {
    schema_key = schema.name();
  } else {
    schema_key = schema.name() + "." + schema.overload_name();
  }

  // First we check if we have a direct mapping onto MLIR.
  auto mlir_handle = _direct_dispatch_lookup.find(schema_key);
  if (mlir_handle != _direct_dispatch_lookup.end()) {
    logging::trace("[TRACING-2] Handling {} via MLIR", schema_key);
    /*
     * Convert the stack arguments into JIT. Only for JIT graph
     * correction/building so we can maintain a JIT graph as well as the MLIR
     * graph.
     */
    // The handler will clear the stack so we collect up all the JIT inputs.
    std::vector<torch::jit::Value *> inputs;
    for (c10::IValue value : (*stack)) {
      if (value.isTensor()) {
        at::Tensor t = value.toTensor();

        // Skip optionals.
        if (!t.defined()) {
          continue;
        }
        // Legality of the JIT doesn't hugely matter, especially for inputs.
        // Each node will at most only look at the output of a previous node,
        // not the input. So we don't think too much here and just skip nodes
        // which we aren't tracking.
        torch::jit::Value *val = _mapper.getValueForTensor(t);
        if (val == nullptr) {
          continue;
        }
        // Add it to the list.
        inputs.push_back(val);
      }
    }

    /*
     * The core MLIR part.
     */
    // Call the handler which empties the stack, calls the MLIR implementation
    // (i.e. builders defined in tablegen), and repopulates the stack.

    // The handler will be found in the compiler dispatch table.
    // See CompilerDispatchTable.cpp AtenToMlirDispatch.inc,
    // AtenToMlirInterface.hpp.inc and AtenToMlirInterface.hpp.inc
    ERROR_ON(
        !_compiler.allOpsCanBeLoweredToPoplar()); // This shouldn't be possible
    mlir_handle->second(*stack);
    ERROR_ON_MSG(!_compiler.allOpsCanBeLoweredToPoplar(),
                 schema_key << " cannot currently be lowered to Poplar");

    /*
     * All logic from here down is to ensure the JIT graph is still correct and
     * legal.
     */
    // The stack contains all the outputs.
    const std::size_t num_outputs = stack->size();
    logging::trace("[TRACING-2] Num outputs: {}", num_outputs);

    // Add a JIT node to keep the jit graph clean.
    torch::jit::Node *node = _graph.create(symbol, inputs, stack->size());
    _graph.appendNode(node);

    // Map the JIT nodes onto the correct outputs.
    for (std::size_t index = 0; index < num_outputs; ++index) {
      at::Tensor tensor = stack->at(index).toTensor();

      if (!tensor.defined()) {
        logging::trace("[TRACING-2] Output: skipping undefined tensor,{}",
                       reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()));
        continue;
      }

      node->output(index)->inferTypeFrom(tensor);
      _mapper.addTensor(tensor, node->output(index));
      logging::trace("[TRACING-2] Output: Tensor ptr {}, jit ir %{} {}",
                     reinterpret_cast<void *>(tensor.unsafeGetTensorImpl()),
                     node->output(index)->debugNameBase(), toString(tensor));
    }

    // Update the 'tail' so if another node needs to process the JIT it will
    // start from here.
    _last_processed_node = node;
  } else {
    logging::trace("[TRACING-2] Handling {} via JIT fallback", schema_key);
    // Otherwise we convert the node into PyTorch JIT first and work with that
    // instead of direct mapping.
    canonicaliseAndLowerViaJit(op, stack);
  }
}

// Convert to JIT and through that lower to MLIR.
void MLIRDispatch::canonicaliseAndLowerViaJit(
    const c10::OperatorHandle &initial_op, c10::Stack *stack) {
  // We can convert any Pytorch schema we see into JIT via the PyTorch JIT API.
  // In the case where we have not mapped an operation on to MLIR or have not
  // directly supported that operation 1:1 in MLIR we can convert that operation
  // to JIT and run through our existing canonicalisation code. That converts
  // from the 1000s of PyTorch ops into the PopART/Onnx op set which we can then
  // lower.

  // A jit node will look like:
  /*
   * graph(...):
   *    aten::node(...)
   */

  // We canonicalise that which will turn it into the PopART/Onnx equivalent.
  /*
   * graph(...):
   *    popart::node(...)
   */
  // We can then just directly add that node as MLIR.

  // OR! The canonicalisation will decompose it into multiple supported popart
  // ops which implements the operation.
  /*
   * graph(...):
   *    popart::SubNode1(...)
   *    popart::SubNode2(...)
   *    popart::SubNode3(...)
   */
  // We then iterate over each of those nodes and add them one by one into MLIR.

  // As we move through the model we build up the MLIR graph AND the JIT graph.
  // To know where to start from we track the last added node.
  /*
   * graph(...):
   *    popart::MatMul(...)
   *    popart::Add(...)        <- _last_processed_node
   *    popart::SubNode1(...)   <- Node we've just added/canonicalised.
   *    popart::SubNode2(...)
   *    popart::SubNode3(...)
   */
  // We can iterate down from that till the end of the graph so we capture all
  // nodes which canonicalisation has added.

  c10::Dispatcher &dispatcher = c10::Dispatcher::singleton();

  const c10::FunctionSchema &initial_schema = initial_op.schema();
  // Run through the schema to find out if one of the operators is supposed to
  // be inplace, this could be the 'out' argument of a non-inplace op.
  c10::intrusive_ptr<at::TensorImpl> inplace_tensor =
      getInplaceArgument(*stack, initial_schema);
  bool is_truly_inplace = isTrulyInplace(*stack, initial_schema);

  c10::OperatorHandle op = initial_op;
  bool op_changed = getOutplaceOpHandle(op, dispatcher);
  const c10::FunctionSchema &schema = op.schema();

  // Create a fake IR node for us to target using the schema.
  torch::jit::Node *node = lowerFromSchema(schema, stack, _graph, _mapper);

  if (shouldRunOnCpu(op_changed || is_truly_inplace, schema.name())) {
    HalfFloatConverter hf_conv(stack, schema, _mapper);
    hf_conv.pre();
    // Call the CPU version to get the output shape
    dispatcher.callBoxed(op, stack);
    hf_conv.post();
  } else {
    // The Op is in place: we don't need to run the CPU version.
    // Just clear the stack and keep the first input.
    at::Tensor t = stack->at(0).toTensor();
    stack->clear();
    stack->push_back(t);
  }

  // Fix the fake tensor so it can still work with our canonicalisation
  // functions which check the output.
  fixNodeOutput(node, *stack, _mapper);

  // Try to canonicalise it.
  torch::jit::Node *new_node =
      canonicalise(schema, node, _graph, true /*is allowed to fail*/);

  // The running output_id. Each operation will decompose into 1 or more onnx
  // operations, we track the last added one via this so we know which one is
  // the "last".
  poptorch_ir::TensorId output_id = poptorch_ir::tensor_error_id;

  // We use the last processed node as the start point for the jit. This is the
  // last node which has been added to the graph, so we can tell what nodes have
  // just been added.
  if (_last_processed_node == nullptr) {
    // We start one behind the first node as when we traverse we are looking a
    // the *next* node as normally we will be starting from the last node,
    // however here the first node is the node we are evaluating.
    _last_processed_node = *(--_graph.nodes().begin());
  }

  torch::jit::Node *start_node = _last_processed_node;

  // When we add a jit node we track the compiler output of that node and map it
  // onto the corresponding JIT value. This is so one expression can refer to
  // its constituent parts. E.G an expression broken into an add and a mul:
  // %3 = mul(%0, %1)
  // %4 = add(%2, %3)
  // This here would track the %3 and %4. We don't need to track them globally
  // because any other expression can't refer to them, only the final node can.
  std::unordered_map<torch::jit::Value *, poptorch_ir::TensorId>
      just_added_nodes;

  for (auto itr = ++start_node->iterator(); itr != _graph.nodes().end();
       ++itr) {
    std::vector<poptorch_ir::TensorId> mlir_ids;
    for (torch::jit::Value *val : itr->inputs()) {
      // See if the value is being tracked globally.
      poptorch_ir::TensorId ssa = _mapper.getMLIRForJit(val);

      // Otherwise check if it is an intermediate produced by this expression.
      if (ssa == poptorch_ir::tensor_error_id || ssa == poptorch_ir::none_id) {
        // Error if we can't find it here.
        auto ssa_itr = just_added_nodes.find(val);
        ERROR_ON_MSG(ssa_itr == just_added_nodes.end(),
                     "JIT node could not be mapped onto MLIR.");
        ssa = ssa_itr->second;
      }

      // Track all the inputs to this operation.
      mlir_ids.push_back(ssa);

      logging::trace("[TRACING-2][JIT] Looking up MLIR for JIT value. ",
                     "Jit name {} jit ptr {} mlir {}", val->debugNameBase(),
                     reinterpret_cast<void *>(val), ssa);
    }

    const c10::Symbol kind = itr->kind();

    // Look up the normal PopART jit node.
    auto jit_handler = _jit_handlers.find(kind);
    if (jit_handler != _jit_handlers.end()) {
      // Call the normal PopART handler.
      output_id = jit_handler->second(*itr, mlir_ids);
    } else if (kind == symbols::popart::identityloss) {
      // We treat identity loss as a special case so we can directly handle the
      // reductions.
      std::vector<std::int64_t> axes = _compiler.getSize(mlir_ids[0]);

      for (std::size_t i = 0; i < axes.size(); ++i) {
        axes[i] = i;
      }

      output_id = _compiler.reducemean(mlir_ids[0], axes, false).at(0).at(0);
    } else {
      const c10::FunctionSchema &notfound_schema = itr->schema();
      ERROR("Could not find any handler for node." + notfound_schema.name());
    }

    just_added_nodes.insert({itr->output(0), output_id});
  }

  // We always want to be looking at "fresh" nodes.
  _last_processed_node = new_node;

  // The output to return
  at::Tensor output;
  auto output_val = stack->at(0);
  ERROR_ON_MSG(!output_val.isTensor(), "Output value is not a tensor.");
  output = output_val.toTensor();

  // If we are inplace copy the result into that tensor.
  if (inplace_tensor) {
    outputIsInplaceOf(output_id, at::Tensor{inplace_tensor});
  } else {
    // Add the tensor + track the jit node.
    _mapper.addTensor(output, new_node->output(0));
    // Track the MLIR node as well.
    _mapper.addTensor(output, output_id);
  }

  // In some cases, cpu shape doesn't match the mlir shape exactly
  // (e.g. scalar vectors). If that's the case, mlir has precedence.
  auto mlir_shape = _compiler.getSize(output_id);
  auto cpu_shape = output.sizes();
  bool shapes_match = mlir_shape.size() == cpu_shape.size();

  for (size_t i = 0; i < mlir_shape.size(); ++i) {
    if (!shapes_match) {
      break;
    }
    if (mlir_shape.at(i) != cpu_shape.at(i)) {
      shapes_match = false;
    }
  }

  if (!shapes_match) {
    output = output.reshape(mlir_shape);
    stack->clear();
    stack->push_back(output);
  }

  // Fixup the JIT so it has the correct type.
  new_node->output(0)->inferTypeFrom(output);

  logging::trace("[TRACING-2][JIT] Output: Tensor ptr {}, jit ir %{} {}",
                 reinterpret_cast<void *>(output.unsafeGetTensorImpl()),
                 new_node->output(0)->debugNameBase(), toString(output));
}

std::shared_ptr<MLIRExecutable> MLIRDispatch::compile() {
  // Get the binary from MLIR.
  poptorch_ir::PoptorchExecutorWrapper executor = _compiler.compile();

  // Print out the timing information about how long each stage takes.
  _compiler.getTimingInfo();

  // Wrap it in a pointer so it can be carried around without needing to leak
  // too much MLIR into the rest of PopTorch.
  auto ptr = std::make_unique<poptorch_ir::PoptorchExecutorWrapper>(
      std::move(executor));

  // Wrap this in a executable shared ptr so we can retain it in pytorch
  // independent of the compiler.
  return std::make_shared<MLIRExecutable>(std::move(ptr));
}

// Resolves a PyTorch tensor to find out what its MLIR representation is.
// Sometimes (i.e when it is a python constant) we will add the missing MLIR.
poptorch_ir::TensorId MLIRDispatch::findTensor(const at::Tensor &tensor) {
  // Undefined tensors are optional tensors which do not exist.
  if (!tensor.defined()) {
    return poptorch_ir::tensor_error_id;
  }

  poptorch_ir::TensorId val = _mapper.getMLIRForTensor(tensor);

  if (val == poptorch_ir::tensor_error_id) {
    // Autograd is annoyingly efficient at "wrapped numbers" from the
    // code. E.G x * 0.8. The 0.8 gets promoted to a constant via a fast
    // path we can't see. This is only because the device is CPU (See
    // ScalarOps.h/cpp) if we had our own device it would go through
    // normal dispatch. Luckily we can still see it is in fact a wrapped number
    // and unpack it safely.

    // They also store and seemingly share a storage location, even though we
    // have prevented them from freeing the tensor. Thats why we check if it is
    // a wrapped number before checking the storage alias map. Once we have our
    // own device wrapped numbers will be explicit copies.
    if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
      ERROR_ON(tensor.numel() != 1);

      // TODO(T49190): More than just doubles and longs.
      switch (tensor.scalar_type()) {
      case c10::ScalarType::Double: {
        std::vector<float> tmp(tensor.numel());
        for (auto i = 0u; i < tensor.numel(); ++i) {
          double wrapped_value = tensor.data_ptr<double>()[i];
          logging::trace("[TRACING-2] Found tensor is a wrapped value: {}",
                         wrapped_value);
          tmp[i] = wrapped_value;
        }
        val = _compiler.tensorconstant_float(tmp).at(0).at(0);
        break;
      }
      case c10::ScalarType::Long: {
        std::vector<std::int32_t> tmp(tensor.numel());
        for (auto i = 0u; i < tensor.numel(); ++i) {
          std::int64_t wrapped_value = tensor.data_ptr<std::int64_t>()[i];
          logging::trace("[TRACING-2] Found tensor is a wrapped value: {}",
                         wrapped_value);
          // Ensure the wrapped value fits in 32 bit
          ERROR_ON_MSG(
              wrapped_value > std::numeric_limits<std::int32_t>::max() ||
                  wrapped_value < std::numeric_limits<std::int32_t>::lowest(),
              "Cannot convert wrapped wrapped integer value ("
                  << wrapped_value
                  << ") to 32-bit signed representation, as the value is "
                     "outside the representable range.");
          tmp[i] = wrapped_value;
        }
        val = _compiler.tensorconstant_int(tmp).at(0).at(0);
        break;
      }
      default:
        ERROR("Wrapped values of scalar type " << tensor.scalar_type()
                                               << " are not supported.");
      }
      _mapper.addTensor(tensor, val);

      logging::trace(
          "[TRACING-2] Added wrapped value tensor, addr: {} with storage {}",
          static_cast<void *>(tensor.unsafeGetTensorImpl()),
          static_cast<void *>(tensor.storage().unsafeGetStorageImpl()));

    } else if (_mapper.isDirectAlias(tensor)) {
      // A value we haven't seen before, we should have captured all tensor
      // creation paths so this should not be reachable normally. However,
      // the autograd can sneakily shallow copy a tensor outwith the
      // dispatch mechanism. So in these cases we do allow it to be an alias
      // if it has the same, shape, storage, and storage offset. Only a
      // direct alias (non-view) is allowed as otherwise we will end up
      // tracking a huge map of tensors and their views.
      val = _mapper.getMLIRForTensor(tensor);
    } else {
      ERROR("\tCould not find tensor " << tensor.data_ptr() << ", "
                                       << toString(tensor) << std::endl);
    }
  }

  return val;
}

at::Tensor MLIRDispatch::outputIsInplaceOf(poptorch_ir::TensorId output_id,
                                           const at::Tensor &original_input) {
  ERROR_ON(output_id == poptorch_ir::none_id ||
           output_id == poptorch_ir::tensor_error_id);

  poptorch_ir::TensorId actual_output = findTensor(original_input);
  _compiler.copy_(actual_output, output_id);
  return original_input;
}

at::Tensor MLIRDispatch::outputInplaceReshape_squeeze_dim_(
    poptorch_ir::TensorId output_id, const at::Tensor &original_input,
    poptorch_ir::TensorId self, int dim) {
  (void)self;
  original_input.squeeze_(dim);
  _mapper.addTensor(original_input, output_id);
  return original_input;
}

at::Tensor MLIRDispatch::makeEmptyOutputTensor(poptorch_ir::TensorId output_id,
                                               bool requires_grad) {
  // If it's a none or error, return an undefined tensor. Some functions may
  // return undefined tensor for certain inputs.
  if (output_id == poptorch_ir::none_id ||
      output_id == poptorch_ir::tensor_error_id) {
    return at::Tensor();
  }

  const std::vector<std::int64_t> shape = _compiler.getSize(output_id);
  poptorch_ir::Type compiler_type = _compiler.getType(output_id);
  auto dtype = compilerTypeToScalarType(compiler_type);
  // Create new tensor
  at::Tensor new_output = at::native::zeros(shape, dtype);
  new_output.set_requires_grad(requires_grad);
  _mapper.addTensor(new_output, output_id);

  return new_output;
}

poptorch_ir::OptionalTensorId MLIRDispatch::getSingleOptionalTensorId(
    const std::vector<poptorch_ir::OptionalTensorId> &tensor_vec) {
  ERROR_ON(tensor_vec.size() > 1);

  if (tensor_vec.empty()) {
    return poptorch_ir::none_id;
  }

  return tensor_vec[0];
}

// A small collection of helpers to help convert PyTorch ATEN into MLIR.

// Some operations are in the form, op(in1, in2, out!) with `out!` not really
// being a true operand to the op but instead is the storage location of the
// output. To handle these cases we check if any of the inputs are part of that
// input.
template <int N>
bool isInplaceOnInput(const at::Tensor &inplaceOutput,
                      const at::Tensor (&inputs)[N]) {
  // Check if the input is the same tensor as the inplace operand.
  for (const at::Tensor &input : inputs) {
    if (inplaceOutput.is_same(input)) {
      return true;
    }
  }

  // The operation is outplace if none of the inputs match.
  return false;
}

inline std::vector<std::int64_t> toIntVector(c10::IValue &value) {
  return value.toIntVector();
}
inline std::int64_t toInt(c10::IValue &value) { return value.toInt(); }

// Use an int vector to avoid the unusual std::vector<bool>: there is also no
// "toBoolVector method."
// (PyTorch uses std::array<bool, N> in at least one place for this.)
inline std::vector<int64_t> toBoolVector(c10::IValue &value) {
  auto bool_list = value.toBoolList();

  std::vector<int64_t> vec;
  std::for_each(bool_list.begin(), bool_list.end(),
                [&vec](bool b) { vec.emplace_back(b); });

  return vec;
}

inline bool toBool(c10::IValue &value) { return value.toBool(); }

inline std::optional<std::int64_t> toOptionalInt(c10::IValue &value) {
  if (value.isNone()) {
    return std::nullopt;
  }
  return value.toInt();
}

inline double toDouble(c10::IValue &value) {
  if (value.isDouble()) {
    return value.toDouble();
  }

  // Fairly common case of `Alpha` being 1
  if (value.isInt()) {
    return static_cast<double>(value.toInt());
  }

  ERROR("Unsupported value type " << value.type()->str() << " in `toDouble`");
}

inline std::optional<double> toOptionalDouble(c10::IValue &value) {
  if (value.isNone()) {
    return std::nullopt;
  }
  if (value.isDouble()) {
    return value.toDouble();
  }
  // Fairly common case of `Alpha` being 1
  if (value.isInt()) {
    return static_cast<double>(value.toInt());
  }

  ERROR("Unsupported value type " << value.type()->str() << " in `toDouble`");
}

#include "AtenToMlirInterface.cpp.inc"

} // namespace poptorch
