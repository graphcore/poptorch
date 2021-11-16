// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "MlirDispatch.hpp"

#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "../../PoptorchSymbols.hpp"
#include "../../popart_canonicalization/PopartCanonicalizationUtils.hpp"
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

static poptorch_ir::Type toCompilerType(const at::Tensor &tensor) {
  at::ScalarType elem_type = tensor.scalar_type();

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

void MLIRDispatch::createGraph(const std::vector<at::Tensor> &inputs,
                               const std::vector<at::Tensor> &parameters) {
  // Init our MLIR compiler.
  _compiler.init();

  // Start timing how long it takes us to build the graph.
  _compiler.startTraceTiming();

  int i = 0;
  // Add any inputs.
  for (const at::Tensor &tensor : inputs) {
    const std::string str = "Input/" + std::to_string(i++);
    std::vector<std::int64_t> shape;
    shape.reserve(tensor.dim());
    for (std::int64_t dim : tensor.sizes()) {
      shape.push_back(dim);
    }

    logging::trace("\tAdding input: {} ",
                   static_cast<void *>(tensor.data_ptr()));
    poptorch_ir::TensorId value = _compiler.addInput(
        tensor.data_ptr(), shape, toCompilerType(tensor), str.c_str());
    _mapper.addTensor(tensor, value);
  }

  i = 0;
  // Add the parameters.
  for (const at::Tensor &tensor : parameters) {
    const std::string str = "Parameter/" + std::to_string(i++);

    std::vector<std::int64_t> shape;
    shape.reserve(tensor.dim());
    for (std::int64_t dim : tensor.sizes()) {
      shape.push_back(dim);
    }

    logging::trace(
        "\tAdding parameter: {} with storage {}",
        static_cast<void *>(tensor.data_ptr()),
        static_cast<void *>(tensor.storage().unsafeGetStorageImpl()));

    poptorch_ir::TensorId value = _compiler.addParameter(
        tensor.data_ptr(), shape, toCompilerType(tensor), str.c_str());
    _mapper.addTensor(tensor, value);
  }
}

void MLIRDispatch::markOutputs(
    const std::vector<at::Tensor> &ids,
    const std::vector<at::Tensor> &persistent_data_storage) {

  ERROR_ON_MSG(
      ids.size() != persistent_data_storage.size(),
      "[INTERNAL] Outputs and output storages do not have same length");

  for (std::size_t i = 0; i < ids.size(); ++i) {
    at::Tensor tensor = ids[i];
    void *storage = persistent_data_storage[i].data_ptr();

    const std::string str = "Output/" + std::to_string(i);
    poptorch_ir::TensorId id = _mapper.getMLIRForTensor(tensor);
    logging::trace(
        "Output: {} with storage {}",
        static_cast<void *>(tensor.unsafeGetTensorImpl()),
        static_cast<void *>(tensor.storage().unsafeGetStorageImpl()));

    if (id != poptorch_ir::tensor_error_id) {
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

at::Tensor &MLIRDispatch::copyInplace(at::Tensor &self, const at::Tensor &src) {
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
    const bool /*transposed*/, const at::IntArrayRef output_padding,
    const int64_t groups) {
  // Create the stack which is just a vector of IValues, I.e all of the above
  // arguments.
  c10::Stack stack;

  // An optional bias is actually an undefined tensor, not 100% why they have
  // two layers of indirection (optional AND undefined).
  packStack(stack, input, weight, *bias, strides, padding, dilation,
            output_padding, groups);

  // Unpack the above and create the MLIR convolution node.
  this->convolution(stack);

  return stack.at(0).toTensor();
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

  // Make an empty IR node.
  poptorch_ir::TensorId out_id = _compiler.empty_tensor(self.sizes().vec());

  // Track it in our mapper.
  _mapper.addTensor(out, out_id);

  // Create a normal copy.
  return copyInplace(out, self);
}

void MLIRDispatch::registerEmptyTensor(const at::Tensor &tensor) {
  // Don't bother intercepting on JIT as we automatically promote unknown nodes
  // to empty tensors.
  poptorch_ir::TensorId id = _compiler.empty_tensor(tensor.sizes().vec());
  _mapper.addTensor(tensor, id);
}

// aten::detach(Tensor(a) self) -> (Tensor(a))
at::Tensor MLIRDispatch::detach(const at::Tensor &self) { return self; }

void MLIRDispatch::fallback(const c10::OperatorHandle &op, c10::Stack *stack) {
  const c10::FunctionSchema &schema = op.schema();

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
    // Call the handler which empties the stack, calls the MLIR, and repopulates
    // the stack.
    mlir_handle->second(*stack);

    // Done.
    return;
  }

  // Otherwise we convert the node into PyTorch JIT first and work with that
  // instead of direct mapping.
  ERROR("Could not find implementation for operation of name: " + schema_key);
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

      // TODO(T49190): More than just doubles.
      std::vector<float> tmp(tensor.numel());

      std::stringstream ss;
      for (std::uint32_t i = 0; i < tensor.numel(); ++i) {
        logging::trace("\tWrapped value: {}", *tensor.data_ptr<double>());
        tmp[i] = tensor.data_ptr<double>()[i];
      }

      val = _compiler.tensorconstant(tmp);
      _mapper.addTensor(tensor, val);

      logging::trace(
          "  addr: {} with storage {}",
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
      std::stringstream stream;
      stream << "\tCould not find tensor with values: [ ";
      for (std::uint32_t i = 0; i < tensor.numel(); ++i) {
        stream << tensor.data_ptr<float>()[i] << ",";
      }
      stream << "]" << std::endl;

      ERROR(stream.str());
    }
  }

  return val;
}

at::Tensor MLIRDispatch::outputIsInplaceOf(poptorch_ir::TensorId output_id,
                                           const at::Tensor &original_input) {
  at::Tensor new_output = at::Tensor(original_input.getIntrusivePtr());
  _mapper.addTensor(new_output, output_id);
  return new_output;
}

at::Tensor MLIRDispatch::makeEmptyOutputTensor(poptorch_ir::TensorId output_id,
                                               bool requires_grad) {
  const std::vector<std::int64_t> shape = _compiler.getSize(output_id);
  // Create new tensor
  at::Tensor new_output = at::native::empty_cpu(shape);
  new_output.set_requires_grad(requires_grad);
  _mapper.addTensor(new_output, output_id);

  return new_output;
}

// If it is a view we need to give it the same storage location as PyTorch
// checks this, at least in debug builds.
at::Tensor MLIRDispatch::outputIsViewOf(poptorch_ir::TensorId output_id,
                                        const at::Tensor &original_input,
                                        bool requires_grad) {
  std::vector<std::int64_t> shape = _compiler.getSize(output_id);

  // Create new tensor
  at::Tensor new_output = at::native::empty_cpu(shape);

  const std::int64_t old_numel = new_output.numel();

  // We need to take the same storage and storage metadata (strides,
  // offsets) as the original.
  new_output = new_output.set_(original_input);

  if (old_numel == new_output.numel()) {
    new_output = new_output.reshape(shape);
  } else {
    // Reshape into the original shape.
    new_output = new_output.reshape(original_input.sizes());

    // Perform the operation as planned.
    new_output = new_output.expand(shape);
  }

  new_output.set_requires_grad(requires_grad);
  _mapper.addTensor(new_output, output_id);

  return new_output;
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

inline bool toBool(c10::IValue &value) { return value.toBool(); }

inline double toDouble(c10::IValue &value) {
  if (value.isDouble()) {
    return value.toDouble();
  }

  // Fairly common case of `Alpha` being 1
  if (value.isInt()) {
    return static_cast<double>(value.toInt());
  }

  ERROR("Unsupported value type in `toDouble`");
}

#include "AtenToMlirInterface.cpp.inc"

} // namespace poptorch
