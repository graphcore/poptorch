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

namespace {
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

void MLIRDispatch::initCompiler() {
  // Init our MLIR compiler.
  _compiler.init();
}

at::Tensor MLIRDispatch::allocateTensorImpl(
    c10::IntArrayRef size, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Device> device, c10::optional<at::Layout> layout,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format) {
  UNUSED(device);

  at::Tensor output = at::native::empty_cpu(size, dtype, layout, device,
                                            pin_memory, memory_format);

  logging::trace("[TRACING-2] Allocated tensor: {} {}",
                 static_cast<void *>(output.unsafeGetTensorImpl()),
                 toString(output));
  return output;
}

at::Tensor MLIRDispatch::addInput(const at::Tensor &cpu_tensor) {
  ERROR_ON(!cpu_tensor.unsafeGetTensorImpl()->is_cpu());

  at::Tensor tensor = allocateTensor(
      cpu_tensor.sizes(), c10::typeMetaToScalarType(cpu_tensor.dtype()));

  const std::string str = "Input/" + std::to_string(_next_input_idx++);
  logging::trace("[TRACING-2] Adding input: {} ",
                 static_cast<void *>(cpu_tensor.unsafeGetTensorImpl()));
  poptorch_ir::TensorId value =
      _compiler.addInput(cpu_tensor.data_ptr(), toCompilerShape(tensor),
                         toCompilerType(tensor), str.c_str());
  _mapper.addTensor(tensor, value);
  // TODO(T59880) Next line shouldn't be needed anymore: we should only see
  // tensors we've allocated ourselves.
  _mapper.addTensor(cpu_tensor, value);
  return tensor;
}

at::Tensor MLIRDispatch::addParameter(const at::Tensor &cpu_tensor) {
  ERROR_ON(!cpu_tensor.unsafeGetTensorImpl()->is_cpu());
  at::Tensor tensor = allocateTensor(
      cpu_tensor.sizes(), c10::typeMetaToScalarType(cpu_tensor.dtype()));

  const std::string str = "Parameter/" + std::to_string(_next_parameter_idx++);

  logging::trace(
      "[TRACING-2] Adding parameter: {} with storage {}", cpu_tensor.data_ptr(),
      static_cast<void *>(cpu_tensor.storage().unsafeGetStorageImpl()));

  poptorch_ir::TensorId value =
      _compiler.addParameter(cpu_tensor.data_ptr(), toCompilerShape(tensor),
                             toCompilerType(tensor), str.c_str());
  _mapper.addTensor(tensor, value);
  // TODO(T59880) Next line shouldn't be needed anymore: we should only see
  // tensors we've allocated ourselves.
  _mapper.addTensor(cpu_tensor, value);
  return tensor;
}

void MLIRDispatch::createGraph() {
  initCompiler();

  // Start timing how long it takes us to build the graph.
  _compiler.startTraceTiming();
}

void MLIRDispatch::addOutput(const at::Tensor &ipu_src,
                             const at::Tensor &cpu_dest) {
  void *storage = cpu_dest.data_ptr();

  const std::string str = "Output/" + std::to_string(_next_output_idx++);
  poptorch_ir::TensorId id = _mapper.getMLIRForTensor(ipu_src);
  logging::trace(
      "[TRACING-2] Marking output: {} with storage {}",
      static_cast<void *>(cpu_dest.unsafeGetTensorImpl()),
      static_cast<void *>(cpu_dest.storage().unsafeGetStorageImpl()));

  if (id != poptorch_ir::tensor_error_id && id != poptorch_ir::none_id) {
    _compiler.addOutput(id, storage, str.c_str());
  }
}

void MLIRDispatch::finalizeGraph() {
  _compiler.addReturn();
  _compiler.endTraceTiming();
}

std::vector<poptorch_ir::TensorId>
MLIRDispatch::mlirFromStack(c10::Stack &stack) {
  std::vector<poptorch_ir::TensorId> ids;

  // For each IValue (which may or may not be a tensor).
  for (c10::IValue value : stack) {
    // Look up the MLIR value if it is a tensor.
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
  // Handle a cast
  // Ignore layout/device/pin/fmt
  if (dtype && (self.dtype() != *dtype)) {
    c10::Stack stack;
    stack.push_back(self);
    stack.push_back(*dtype);
    to_dtype(stack);
    return stack.at(0).toTensor();
  }

  // Create the new output tensor.
  at::Tensor out =
      allocateTensor(self.sizes(), dtype, device, layout, pin, fmt);

  // Zero tensor as it's possible that the tensor is accessible by the user
  // after tracing.
  at::zero_(out);

  // Make an empty IR node.
  // (empty_tensor returns an ODSTensorResults instance, which is a vector of a
  // vector of a tensor)
  poptorch_ir::TensorId out_id =
      _compiler
          .empty_tensor(self.sizes().vec(),
                        toCompilerType(scalarTypeOrDefault(dtype)))
          .at(0)
          .at(0);

  // Track it in our mapper.
  _mapper.addTensor(out, out_id);

  // Create a normal copy in the graph (this does not alter the at::tensor)
  return copyInplace(out, self);
}

void MLIRDispatch::registerEmptyTensor(const at::Tensor &tensor) {
  poptorch_ir::TensorId id =
      _compiler.empty_tensor(tensor.sizes().vec(), toCompilerType(tensor))
          .at(0)
          .at(0);
  _mapper.addTensor(tensor, id);
}

// aten::detach(Tensor(a) self) -> (Tensor(a))
at::Tensor MLIRDispatch::detach(const at::Tensor &self) {
  return toCopyInplace(self, c10::nullopt, c10::nullopt, self.device());
}

void MLIRDispatch::setCurrentCodeLocation(
    const torch::jit::SourceRange &source_location) {
  auto file_line_col = source_location.file_line_col();
  if (file_line_col) {
    std::uint64_t line;
    std::uint64_t col;
    std::string filename;
    std::tie(filename, line, col) = *file_line_col;
    _compiler.setCurrentPythonCodeLocation(filename.c_str(), line, col);
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
  ERROR_ON_MSG(mlir_handle == _direct_dispatch_lookup.end(),
               "No shape inference handler for " << schema_key);
  logging::trace("[TRACING-2] Handling {} via MLIR", schema_key);

  /*
   * The core MLIR part.
   */
  // Call the handler which empties the stack, calls the MLIR implementation
  // (i.e. builders defined in tablegen), and repopulates the stack.

  // The handler will be found in the compiler dispatch table.
  // See CompilerDispatchTable.cpp AtenToMlirDispatch.inc,
  // AtenToMlirInterface.hpp.inc and AtenToMlirInterface.hpp.inc
  mlir_handle->second(*stack);
  return schema_key;
}

void MLIRDispatch::fallback(const c10::OperatorHandle &op, c10::Stack *stack) {
  ERROR_ON(
      !_compiler.allOpsCanBeLoweredToPoplar()); // This shouldn't be possible
  const std::string schema_key = handleOp(op, stack);
  ERROR_ON_MSG(!_compiler.allOpsCanBeLoweredToPoplar(),
               schema_key << " cannot currently be lowered to Poplar");
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
      ERROR("\tCould not find tensor " << tensor.unsafeGetTensorImpl() << ", "
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
  at::Tensor new_output = allocateTensor(shape, dtype);
  // Zero tensor as it's possible that the tensor is accessible by the user
  // after tracing.
  at::zero_(new_output);
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

inline poptorch_ir::Type toCompilerType(c10::IValue &value) {
  return toCompilerType(value.toScalarType());
}

inline std::optional<poptorch_ir::Type>
toOptionalCompilerType(c10::IValue &value) {
  if (value.isNone()) {
    return std::nullopt;
  }
  return toCompilerType(value.toScalarType());
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
