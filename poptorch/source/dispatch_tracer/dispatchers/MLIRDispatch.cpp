// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "MLIRDispatch.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "../../PoptorchSymbols.hpp"
#include "../../popart_canonicalization/PopartCanonicalizationUtils.hpp"
#include "MLIRDispatchUtils.hpp"
#include "poptorch/DispatchTracer.hpp"
#include "poptorch/OpBuilder.hpp"
#include "poptorch/PopartCanonicalization.hpp"
#include "poptorch/TypeAndConstantCanonicalization.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_err/ExceptionHandling.hpp"
#include "poptorch_logging/Logging.hpp"

#include "pytorch_bridge/CompilerOptions.hpp"

#include "../CommonHelperFunctions.hpp"
#include "../Tensor.hpp"
#include "pytorch_bridge/CompilerTypes.hpp"

namespace poptorch {

namespace {

// Sets requires_grad=true on a tensor iff it is a floating-point type.
// Other types (except complex unsupported on IPU) cannot have requires_grad
// set to true.
void setRequiresGradIfFloat(const at::Tensor &tensor) {
  if (c10::isFloatingType(tensor.scalar_type())) {
    tensor.set_requires_grad(true);
  }
}

poptorch_ir::Type toCompilerType(const at::ScalarType &elem_type) {
  switch (elem_type) {
  case at::ScalarType::Bool:
    return poptorch_ir::Type::BOOL;
  case at::ScalarType::Byte:
    return poptorch_ir::Type::UNSIGNED_CHAR;
  case at::ScalarType::Char:
    return poptorch_ir::Type::CHAR;
  case at::ScalarType::Float:
  case at::ScalarType::Double: // We will convert this.
    return poptorch_ir::Type::FLOAT;
  case at::ScalarType::Half:
    return poptorch_ir::Type::HALF;
  case at::ScalarType::Short:
    return poptorch_ir::Type::SHORT;
  case at::ScalarType::Int:
  case at::ScalarType::Long: // We will convert this.
    return poptorch_ir::Type::INT;
  default:
    ERROR("Unsupported tensor input type from pytorch: " << elem_type);
  }
}

poptorch_ir::Type toCompilerType(const at::Tensor &tensor) {
  at::ScalarType elem_type = tensor.scalar_type();

  return toCompilerType(elem_type);
}

std::vector<std::int64_t> toCompilerShape(const at::Tensor &tensor) {
  std::vector<std::int64_t> shape;
  shape.reserve(tensor.dim());
  for (std::int64_t dim : tensor.sizes()) {
    shape.push_back(dim);
  }
  return shape;
}

} // namespace

MLIRExecutor::MLIRExecutor(
    std::unique_ptr<poptorch_ir::PoplarExecutorWrapper> &&other)
    : _executor(std::move(other)) {}

MLIRExecutor::~MLIRExecutor() {
  // Once we load another executable on to the device, we lose the data
  // associated with it, such as weights, so we need to copy them off the
  // device if the executor is to be swapped out and those host-side buffers
  // have not already been updated since the last call to execute().
  copyWeightsToHostIfNeeded();
}

std::vector<at::Tensor>
MLIRExecutor::execute(const std::vector<at::Tensor> &inputs) {
  try {
    std::vector<void *> input_ptrs;
    input_ptrs.resize(inputs.size());

    // Keep the refs around.
    std::vector<at::Tensor> converted;

    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const at::Tensor &tensor = inputs[i];
      // TODO(T59880) rename is_xla() -> is_ipu()
      if (tensor.is_xla()) {
        // This is the tensor we used to compile the executable: the pointer
        // was already set when _compiler.addInput() was called.
        const auto &host_buffer = getHostBuffer(tensor);
        ERROR_ON_MSG(!host_buffer,
                     "Attempted to pass an IPU tensor with no `host_buffer`.");
        input_ptrs[i] = host_buffer->data();
      } else {
        ERROR_ON_MSG(!tensor.is_cpu(),
                     "Input tensors expected to either be IPU or CPU but input "
                         << i << " is of type "
                         << tensor.unsafeGetTensorImpl()->device_type());
        // Handle the implicit downcasts here:
        if (tensor.scalar_type() == at::ScalarType::Long) {
          converted.push_back(tensor.to(at::ScalarType::Int));
          input_ptrs[i] = converted.back().data_ptr();
        } else if (tensor.scalar_type() == at::ScalarType::Double) {
          converted.push_back(tensor.to(at::ScalarType::Float));
          input_ptrs[i] = converted.back().data_ptr();
        } else {
          input_ptrs[i] = tensor.data_ptr();
        }
      }
    }

    std::vector<at::Tensor> outputs;
    for (const auto &[dims, type] : _executor->outputTypes()) {
      outputs.push_back(
          at::empty(dims, at::dtype(compilerTypeToScalarType(type))
                              .memory_format(c10::MemoryFormat::Contiguous)));
    }
    std::vector<void *> output_ptrs;
    std::transform(outputs.begin(), outputs.end(),
                   std::back_inserter(output_ptrs),
                   [](const at::Tensor &tensor) { return tensor.data_ptr(); });

    // Mark any host buffers as invalid. This can be made much more granular, to
    // each tensor, if required.
    _host_buffers_are_dirty = true;

    _executor->execute(input_ptrs, output_ptrs);

    return outputs;
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

void MLIRExecutor::weightsToDevice() {
  try {
    ERROR_ON(_host_buffers_are_dirty);
    _executor->weightsToDevice();
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
}

void MLIRExecutor::copyWeightsToHostIfNeeded() {
  if (!_host_buffers_are_dirty) {
    poptorch::logging::trace("Ignored copyWeightsToHost: not needed");
    return;
  }

  try {
    _executor->weightsToHost();
  }
  CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION

  // The host weights have now been updated since last execution.
  _host_buffers_are_dirty = false;
}

MLIRDispatch::MLIRDispatch(CompilerOptions const &options,
                           TensorStore *tensor_store)
    : IDispatch(tensor_store) {
  this->generateDispatchTable();
  initCompiler(options);

  // Start timing how long it takes us to build the graph.
  _compiler.startTraceTiming();
}

void MLIRDispatch::initCompiler(const CompilerOptions &options) {
  // Init our MLIR compiler.
  if (options.eager.eager_mode) {
    _compiler.init(poptorch_ir::ExecutionType::EagerMode,
                   poptorch_ir::CompilerBackend::Poplar, options);
  } else {
    _compiler.init(poptorch_ir::ExecutionType::StaticGraph,
                   poptorch_ir::CompilerBackend::Poplar, options);
  }
}

at::Tensor MLIRDispatch::addConstant(const at::Tensor &cpu_tensor) {
  ERROR_ON(!cpu_tensor.unsafeGetTensorImpl()->is_cpu());
  if (isEagerMode()) {
    // Everything is considered an input in eager mode.
    return addInput(cpu_tensor);
  }

  const auto cpu_dtype = cpu_tensor.scalar_type();

  // Convert the CPU tensor's dtype to `int` or `float`, for ease of adding.
  at::Tensor coerced_cpu_tensor = cpu_tensor;
  if (c10::isIntegralType(cpu_dtype, true) &&
      cpu_dtype != c10::ScalarType::Int) {
    coerced_cpu_tensor = coerced_cpu_tensor.to(c10::ScalarType::Int);
  } else if (c10::isFloatingType(cpu_dtype) &&
             cpu_dtype != c10::ScalarType::Float) {
    coerced_cpu_tensor = coerced_cpu_tensor.to(c10::ScalarType::Float);
  }

  if (cpu_dtype == c10::ScalarType::Long ||
      cpu_dtype == c10::ScalarType::Double) {
    logging::warn("Constant tensor (shape {}) coerced from {} to {}",
                  cpu_tensor.sizes(), cpu_dtype,
                  coerced_cpu_tensor.scalar_type());
  }

  // Create the IPU tensor constant.
  at::Tensor tensor = _tensor_store->allocateTensor(
      coerced_cpu_tensor.sizes(), coerced_cpu_tensor.scalar_type());

  const auto shape = tensor.sizes().vec();

  poptorch_ir::TensorId val;
  if (coerced_cpu_tensor.scalar_type() == c10::ScalarType::Float) {
    const auto *data = static_cast<float *>(coerced_cpu_tensor.data_ptr());
    const std::vector<float> tmp(data, data + coerced_cpu_tensor.numel());
    val = _compiler.tensorconstant_float(tmp, shape).at(0).tensor_ids.at(0);
  } else if (coerced_cpu_tensor.scalar_type() == c10::ScalarType::Int) {
    const auto *data = static_cast<int32_t *>(coerced_cpu_tensor.data_ptr());
    const std::vector<int32_t> tmp(data, data + coerced_cpu_tensor.numel());
    val = _compiler.tensorconstant_int(tmp, shape).at(0).tensor_ids.at(0);
  } else {
    ERROR("Tensor constants of scalar type " << tensor.scalar_type()
                                             << " are not supported.");
  }

  // If we've upcast a smaller dtype, cast it back down.
  bool should_recast =
      (c10::isIntegralType(cpu_dtype, true) &&
       cpu_dtype != c10::ScalarType::Int && cpu_dtype != c10::ScalarType::Long);
  should_recast |=
      (c10::isFloatingType(cpu_dtype) && cpu_dtype != c10::ScalarType::Float &&
       cpu_dtype != c10::ScalarType::Double);
  if (should_recast) {
    val = _compiler.cast(val, toCompilerType(cpu_dtype)).at(0).tensor_ids.at(0);
  }

  logging::trace("[DISPATCHER] Adding constant: {} with cpu ptr {}",
                 static_cast<void *>(cpu_tensor.unsafeGetTensorImpl()),
                 cpu_tensor.data_ptr());

  _mapper.addTensor(tensor, val);
  return tensor;
}

void MLIRDispatch::promoteAsParameter(at::Tensor &tensor) {
  ERROR_ON(!isIpuTensor(tensor));
  const std::string str = "Parameter/" + std::to_string(_next_parameter_idx++);
  poptorch_ir::TensorId value = _compiler.addParameter(
      poptorch::getHostBuffer(tensor), toCompilerShape(tensor),
      toCompilerType(tensor), str.c_str());
  _mapper.addTensor(tensor, value);
  setIsParameter(tensor, true);
}

void MLIRDispatch::promoteAsInput(at::Tensor &tensor, bool is_wrapped) {
  if (!isIpuTensor(tensor)) {
    ERROR_ON_MSG(is_wrapped, "All inputs to an `ipu_wrapper`-wrapped function "
                             "must be an IPU tensor, but got an input with "
                             "device="
                                 << tensor.device().type() << '.');
    ERROR("Attempted to promote a CPU tensor as an input to an IPU model.");
  }
  const std::string str = "Input/" + std::to_string(_next_input_idx++);
  poptorch_ir::TensorId value = _compiler.addInput(
      poptorch::getHostBuffer(tensor), toCompilerShape(tensor),
      toCompilerType(tensor), str.c_str());
  _mapper.addTensor(tensor, value);
  setIsParameter(tensor, false);
}

at::Tensor MLIRDispatch::addInput(const at::Tensor &cpu_tensor) {
  logging::trace("[DISPATCHER] Adding input: {} with cpu ptr {}",
                 static_cast<void *>(cpu_tensor.unsafeGetTensorImpl()),
                 cpu_tensor.data_ptr());
  at::Tensor tensor = _tensor_store->copyCpuTensorAsIpuTensor(cpu_tensor);

  promoteAsInput(tensor);

  return tensor;
}

at::Tensor MLIRDispatch::addParameter(const at::Tensor &cpu_tensor) {
  ERROR_ON(!cpu_tensor.unsafeGetTensorImpl()->is_cpu());
  at::Tensor tensor = _tensor_store->allocateTensor(
      cpu_tensor.sizes(), c10::typeMetaToScalarType(cpu_tensor.dtype()));

  const std::string str = "Parameter/" + std::to_string(_next_parameter_idx++);

  logging::trace("[DISPATCHER] Adding parameter: {} with cpu ptr {}",
                 static_cast<void *>(cpu_tensor.unsafeGetTensorImpl()),
                 cpu_tensor.data_ptr());

  copyDataFromCpuSource(tensor, cpu_tensor);
  poptorch_ir::TensorId value = _compiler.addParameter(
      poptorch::getHostBuffer(tensor), toCompilerShape(tensor),
      toCompilerType(tensor), str.c_str());
  _mapper.addTensor(tensor, value);
  setIsParameter(tensor, true);
  return tensor;
}

void MLIRDispatch::promoteAsOutput(const at::Tensor &tensor, void *storage) {
  const std::string str = "Output/" + std::to_string(_next_output_idx++);
  poptorch_ir::TensorId id = _mapper.getMLIRForTensor(tensor);
  logging::trace("[DISPATCHER] Marking output: {} with storage ptr {}",
                 static_cast<void *>(tensor.unsafeGetTensorImpl()), storage);

  if (id != poptorch_ir::tensor_error_id && id != poptorch_ir::none_id) {
    _compiler.addOutput(id, storage, str.c_str());
  }
}

void MLIRDispatch::addOutput(const at::Tensor &ipu_src,
                             const at::Tensor &cpu_dest) {
  void *storage = cpu_dest.data_ptr();
  promoteAsOutput(ipu_src, storage);
}

void MLIRDispatch::finalizeGraph() {
  _compiler.addReturn();
  _compiler.endTraceTiming();
}

void MLIRDispatch::markStep() { _compiler.compileRunAndReset(); }

const at::Tensor &MLIRDispatch::copyInplace(const at::Tensor &self,
                                            const at::Tensor &src) {
  auto mlir_output = _compiler.copy_(findTensor(self), findTensor(src));

  std::vector<bool> requires_grad;
  requires_grad = requiresGrad(mlir_output[0].requires_grad_types,
                               self.requires_grad() || src.requires_grad());

  outputIsInplaceOf(mlir_output[0].tensor_ids[0], self, requires_grad[0]);

  return self;
}

void packStack(c10::Stack & /*unused*/) {}

// A small helper to populate the c10::stack.
template <typename T, typename... Args>
void packStack(c10::Stack &stack, T &arg, Args... args) {
  stack.push_back(arg);
  packStack(stack, args...);
}

void MLIRDispatch::registerEmptyTensor(const at::Tensor &tensor) {
  poptorch_ir::TensorId id =
      _compiler.empty_tensor(tensor.sizes().vec(), toCompilerType(tensor))
          .at(0)
          .tensor_ids.at(0);
  _mapper.addTensor(tensor, id);
}

// aten::detach(Tensor(a) self) -> (Tensor(a))
void MLIRDispatch::detach(const c10::OperatorHandle &op, c10::Stack *stack,
                          bool /*moving_parameters*/) {
  const c10::FunctionSchema &schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  const auto arguments = torch::jit::last(stack, num_arguments);

  ERROR_ON(arguments.size() != 1);
  at::Tensor in = arguments.front().toTensor();

  at::Tensor out(in.unsafeGetTensorImpl()->shallow_copy_and_detach(
      /*version_counter=*/in.unsafeGetTensorImpl()->version_counter(),
      /*allow_tensor_metadata_change=*/true));

  // The new tensor points at the same mlir tensor as the source.
  _mapper.addTensor(out, findTensor(in));

  torch::jit::drop(stack, num_arguments);
  torch::jit::push(stack, out);
}

const std::vector<std::vector<char>> &
MLIRDispatch::getSourceLocationExcludes() const {
#if POPTORCH_BUILD_MLIR_COMPILER
  return _compiler.getOptions().dispatcher.source_location_excludes;
#else
  return {};
#endif
}

void MLIRDispatch::setCurrentCodeLocation(
    const torch::jit::SourceRange &source_location) {
  auto file_line_col = source_location.file_line_col();
  if (file_line_col) {
    auto [filename, line, col] = *file_line_col; // NOLINT
    _compiler.setCurrentPythonCodeLocation(filename.c_str(), line, col);
  } else {
    _compiler.setCurrentPythonCodeLocation(nullptr, 0, 0);
  }
}

std::string MLIRDispatch::handleOp(const c10::OperatorHandle &op,
                                   c10::Stack *stack) {
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
  if (mlir_handle == _direct_dispatch_lookup.end()) {
    std::string s = "No shape inference handler for " + schema_key;
    // In some cases Torch will crash during the exception handling
    // so print the error message on the error channel before throwing
    // the exception.
    logging::err("{}", s);
    ERROR(s);
  }
  logging::trace("[DISPATCHER] Handling {} via MLIR", schema_key);

  /*
   * The core MLIR part.
   */
  // Call the handler which empties the stack, calls the MLIR implementation
  // (i.e. builders defined in tablegen), and repopulates the stack.

  // The handler will be found in the compiler dispatch table.
  // See CompilerDispatchTable.cpp, {Aten|Poptorch}ToMLIRDispatch.inc,
  // {Aten|Poptorch}ToMLIRInterface.hpp.inc and
  // {Aten|Poptorch}ToMLIRInterface.cpp.inc
  mlir_handle->second(*stack);
  return schema_key;
}

void MLIRDispatch::findAndPromoteExternalTensors(c10::Stack *stack) {
  for (auto &value : *stack) {
    if (!value.isTensor()) {
      continue;
    }

    auto &tensor = value.toTensor();

    if (isIpuTensor(tensor) &&
        _mapper.getMLIRForTensor(tensor) == poptorch_ir::tensor_error_id) {
      if (const auto original = getTensorDetails(tensor)->alias_of) {
        _aliases_to_restore.push_back(getTensorDetails(tensor).get());
      }

      promoteAsParameter(tensor);
    }
  }
}

void MLIRDispatch::fallback(const c10::OperatorHandle &op, c10::Stack *stack) {
  ERROR_ON(
      !_compiler.allOpsCanBeLoweredToPoplar()); // This shouldn't be possible

  // Find any tensors which were created before this Dispatch was created, and
  // promote them by marking them as parameters to the graph.
  findAndPromoteExternalTensors(stack);

  const std::string schema_key = handleOp(op, stack);

  ERROR_ON_MSG(!_compiler.allOpsCanBeLoweredToPoplar(),
               schema_key << " cannot currently be lowered to Poplar");
  // Let the compiler know we added a new op to the graph.
  _compiler.onOpAdded();
}

void MLIRDispatch::restoreAliases() {
  for (auto *details : _aliases_to_restore) {
    auto *orig_details = _mapper.getTensorDetailsForId(*details->alias_of);

    _mapper.aliasTensor(details, orig_details);
  }

  _aliases_to_restore.clear();
}

std::shared_ptr<MLIRExecutor> MLIRDispatch::compile() {
  // Restore any aliases we found when promoting tensors which have not already
  // been tied together in the ValueMapper.
  restoreAliases();

  // Get the binary from MLIR.
  poptorch_ir::PoplarExecutorWrapper executor = _compiler.compileAndLoad();

  // Print out the timing information about how long each stage takes.
  _compiler.getTimingInfo();

  // Wrap it in a pointer so it can be carried around without needing to leak
  // too much MLIR into the rest of PopTorch.
  auto ptr =
      std::make_unique<poptorch_ir::PoplarExecutorWrapper>(std::move(executor));

  // Wrap this in a shared pointer so we can retain it in PyTorch independent of
  // the compiler.
  return std::make_shared<MLIRExecutor>(std::move(ptr));
}

// Resolves a PyTorch tensor to find out what its MLIR representation is.
// Sometimes (i.e when it is a python constant) we will add the missing MLIR.
poptorch_ir::TensorId MLIRDispatch::findTensor(const at::Tensor &tensor) {
  // Undefined tensors are optional tensors which do not exist. Note that
  // these are not the same as None IValue types. For example, they appear
  // in lists of optional tensors (Tensor?[]), which must only contain
  // tensor types.
  if (!tensor.defined()) {
    return poptorch_ir::none_id;
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
          logging::trace("[DISPATCHER] Found tensor is a wrapped value: {}",
                         wrapped_value);
          tmp[i] = wrapped_value;
        }
        val = _compiler.tensorconstant_float(tmp, {}).at(0).tensor_ids.at(0);
        break;
      }
      case c10::ScalarType::Long: {
        std::vector<std::int32_t> tmp(tensor.numel());
        for (auto i = 0u; i < tensor.numel(); ++i) {
          std::int64_t wrapped_value = tensor.data_ptr<std::int64_t>()[i];
          logging::trace("[DISPATCHER] Found tensor is a wrapped value: {}",
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
        val = _compiler.tensorconstant_int(tmp, {}).at(0).tensor_ids.at(0);
        break;
      }
      default:
        ERROR("Wrapped values of scalar type " << tensor.scalar_type()
                                               << " are not supported.");
      }
      // Don't track constants in the ValueMapper: they're CPU tensors.

      logging::trace(
          "[DISPATCHER] Added wrapped value tensor, addr: {} with storage {}",
          static_cast<void *>(tensor.unsafeGetTensorImpl()),
          static_cast<void *>(tensor.storage().unsafeGetStorageImpl()));

    } else {
      if (!isIpuTensor(tensor)) {
        ERROR("Expected an IPU tensor but got tensor(device="
              << tensor.unsafeGetTensorImpl()->device_type()
              << ", shape=" << tensor.unsafeGetTensorImpl()->sizes()
              << ", dtype= " << tensor.unsafeGetTensorImpl()->dtype()
              << ").\nConstant tensors should be moved explicitly to the IPU, "
                 "via cpu_tensor.to(ipu_tensor.device).");
      } else {
        ERROR("Could not find tensor " << str(tensor) << std::endl);
      }
    }
  }

  return val;
}

at::Tensor
MLIRDispatch::outputIsInplaceOf(poptorch_ir::OptionalTensorId output_id,
                                const at::Tensor &original_input,
                                bool requires_grad) {
  ERROR_ON(output_id == poptorch_ir::none_id ||
           output_id == poptorch_ir::tensor_error_id);

  const auto original_shape = original_input.sizes();
  const auto new_shape = _compiler.getSize(output_id);

  if (original_shape != new_shape) {
    // There are two situations where the original and new shapes may not match.
    // This will cause issue with mismatched shapes if the mlir::Value for the
    // tensor is not replaced.
    //
    // Firstly, for PyTorch does not always generate outputs of the correct
    // shape. For example,
    //   def f():
    //     x = torch.logicalAnd(torch.tensor(True), torch.tensor(False))
    //     return x
    // will add an implicit output parameter with shape [0] rather than an empty
    // shape. Leading to something like:
    //   func f(%x) {
    //     %0 = empty(shape=[0])
    //     %x = logicalAnd(tensor(true), tensor(false), %0)
    //     return %x
    //   }
    // Note that hiding the implicitly created output parameter is not an issue
    // since it cannot be a view tensor.
    //
    // Secondly, if the source of the output is a view operation, all subsequent
    // references to the original input should be replaced by the view. For
    // example,
    //   def f(x):
    //     y = x + 1
    //     x.select_(...)
    //     z = x + 1
    // If we inserted an overwrite op in the to handle `select_`, we would have
    // no globally consistent shape for x. Instead this is lowered to
    //   func f(%x) {
    //     %y = add(%x, 1)
    //     %0 = select(%x, ...)
    //     %z = add(%0, 1)
    //   }
    // Replacing the original tensor isn't an issue when handling view
    // operations since the original view of x cannot be referenced after the
    // inplace operation.
    _mapper.addTensor(original_input, output_id);
  } else {
    // Instead of replacing all subsequent references to the original input with
    // the tensor we add an overwrite operation (which will cause the input to
    // be replaced later) this makes implementing the view operations easier
    poptorch_ir::TensorId replaced_id = findTensor(original_input);
    _compiler.overwrite(replaced_id, output_id);
  }
  original_input.unsafeGetTensorImpl()->set_sizes_contiguous(new_shape);

  if (requires_grad) {
    setRequiresGradIfFloat(original_input);
  }

  return original_input;
}

std::vector<at::Tensor> MLIRDispatch::outputIsInplaceOfList(
    const std::vector<poptorch_ir::OptionalTensorId> &output_id,
    const std::vector<at::Tensor> &original_input,
    const std::vector<bool> &requires_grad) {
  for (size_t i = 0; i < output_id.size(); i++) {
    outputIsInplaceOf(output_id[i], original_input[i], requires_grad[i]);
  }
  return original_input;
}

std::vector<bool> MLIRDispatch::requiresGrad(
    const std::vector<poptorch_ir::RequiresGradType> &requires_grad_types,
    bool requires_grad_or) {
  std::vector<bool> result(requires_grad_types.size());
  for (size_t i = 0; i < requires_grad_types.size(); i++) {
    switch (requires_grad_types[i]) {
    case poptorch_ir::RequiresGradType::OR_INPUTS:
      result[i] = requires_grad_or;
      break;
    case poptorch_ir::RequiresGradType::FALSE:
      result[i] = false;
      break;
    }
  }
  return result;
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
  at::Tensor new_output = _tensor_store->allocateTensor(shape, dtype);

  if (requires_grad) {
    setRequiresGradIfFloat(new_output);
  }
  _mapper.addTensor(new_output, output_id);

  return new_output;
}

std::vector<at::Tensor> MLIRDispatch::makeEmptyOutputTensorList(
    const std::vector<poptorch_ir::OptionalTensorId> &output_ids,
    const std::vector<bool> &requires_grad) {
  std::vector<at::Tensor> output;
  output.reserve(output_ids.size());
  for (size_t i = 0; i < output_ids.size(); i++) {
    output.push_back(
        makeEmptyOutputTensor(output_ids.at(i), requires_grad.at(i)));
  }
  return output;
}

CompilerOptions &MLIRDispatch::getMutableCompilerOptions() {
#if POPTORCH_BUILD_MLIR_COMPILER
  return _compiler.getMutableOptions();
#else
  ERROR("PopTorch must be compiled with POPTORCH_BUILD_MLIR_COMPILER=ON");
#endif
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

namespace {
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

std::vector<std::int64_t> toIntVector(c10::IValue &value) {
  return value.toIntVector();
}

std::optional<std::vector<std::int64_t>>
toOptionalIntVector(c10::IValue &value) {
  if (value.isNone()) {
    return std::nullopt;
  }
  return value.toIntVector();
}

std::int64_t toInt(c10::IValue &value) { return value.toInt(); }

// Use an int vector to avoid the unusual std::vector<bool>: there is also no
// "toBoolVector method."
// (PyTorch uses std::array<bool, N> in at least one place for this.)
std::vector<int64_t> toBoolVector(c10::IValue &value) {
  auto bool_list = value.toBoolList();

  std::vector<int64_t> vec;
  std::for_each(bool_list.begin(), bool_list.end(),
                [&vec](bool b) { vec.emplace_back(b); });

  return vec;
}

std::vector<at::Tensor> toTensorVector(c10::IValue &value,
                                       const std::string &op) {
  if (value.isTensorList()) {
    return value.toTensorVector();
  }
  ERROR_ON_MSG(!value.isList(), "Expected TensorList or GenericList for "
                                    << op << " but got " << value.tagKind());
  std::vector<at::Tensor> tensors;
  auto tensor_list = value.toList();
  tensors.reserve(tensor_list.size());
  for (c10::IValue v : tensor_list) {
    ERROR_ON_MSG(!v.isTensor(), "Expected a list of tensors for "
                                    << op << " but found a " << v.tagKind()
                                    << " in list");
    tensors.push_back(v.toTensor());
  }
  return tensors;
}

bool toBool(c10::IValue &value) { return value.toBool(); }

std::optional<std::int64_t> toOptionalInt(c10::IValue &value) {
  if (value.isNone()) {
    return std::nullopt;
  }
  return value.toInt();
}

double toDouble(c10::IValue &value) {
  if (value.isDouble()) {
    return value.toDouble();
  }

  // Fairly common case of `Alpha` being 1
  if (value.isInt()) {
    return static_cast<double>(value.toInt());
  }

  ERROR("Unsupported value type " << value.type()->str() << " in `toDouble`");
}

std::vector<float> toFloatVector(c10::IValue &value) {
  if (value.isDoubleList()) {
    auto dv = value.toDoubleVector();
    return std::vector<float>(std::begin(dv), std::end(dv));
  }
  if (value.isIntList()) {
    auto int_vec = value.toIntVector();
    std::vector<float> ret;
    ret.reserve(int_vec.size());
    std::transform(int_vec.begin(), int_vec.end(), std::back_inserter(ret),
                   [](std::int64_t val) { return static_cast<double>(val); });
    return ret;
  }
  ERROR("Unsupported value type " << value.type()->str()
                                  << " in `toDoubleVector`");
}

std::optional<std::vector<float>> toOptionalFloatVector(c10::IValue &value) {
  if (value.isNone()) {
    return std::nullopt;
  }
  return toFloatVector(value);
}

const char *toStr(c10::IValue &value) { return value.toStringRef().c_str(); }

std::optional<const char *> toOptionalStr(c10::IValue &value) {
  if (value.isNone()) {
    return std::nullopt;
  }
  return toStr(value);
}

poptorch_ir::Type toCompilerType(c10::IValue &value) {
  return toCompilerType(value.toScalarType());
}

std::optional<poptorch_ir::Type> toOptionalCompilerType(c10::IValue &value) {
  if (value.isNone()) {
    return std::nullopt;
  }
  return toCompilerType(value.toScalarType());
}

std::optional<double> toOptionalDouble(c10::IValue &value) {
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

} // namespace

#include "AtenToMLIRInterface.cpp.inc"
#include "PoptorchToMLIRInterface.cpp.inc"
#include "TorchScatterToMLIRInterface.cpp.inc"

bool MLIRDispatch::isEagerMode() const {
  return _compiler.getOptions().eager.eager_mode;
}
} // namespace poptorch
