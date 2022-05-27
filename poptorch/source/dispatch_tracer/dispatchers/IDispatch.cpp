// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "IDispatch.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "../CommonHelperFunctions.hpp"
#include "../Tensor.hpp"
#include "poptorch/Utils.hpp"

namespace poptorch {

at::Tensor IDispatch::allocateTensor(
    c10::IntArrayRef size, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Device> device, c10::optional<at::Layout> layout,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format) {
  UNUSED(layout);
  UNUSED(memory_format);
  UNUSED(pin_memory);
  at::ScalarType scalar_type = scalarTypeOrDefault(dtype);
  auto coerced_scalar_type = coerceToSupportedType(scalar_type);

  at::Tensor output = createIpuTensor(
      coerced_scalar_type, deviceOrDefaultIpu(device), _next_tensor_id++, size,
      at::detail::defaultStrides(size));
  if (scalar_type != coerced_scalar_type) {
    logging::warn(
        "[TRACING-2] Allocated tensor: {} {} (type coerced from {} to {})",
        ipuTensorId(output), toString(output), scalar_type,
        coerced_scalar_type);
  } else {
    logging::trace("[TRACING-2] Allocated tensor: {} {}", ipuTensorId(output),
                   toString(output));
  }
  return output;
}

void *IDispatch::getDataSource(torch::jit::Value *value) {
  auto *record = _mapper.rawTensorRecord(value);
  if (record == nullptr) {
    logging::trace("JIT value not tracked {}", reinterpret_cast<void *>(value));
    return nullptr;
  }
  return getCpuData(*record->tensor_impl)->data();
}

bool IDispatch::isParameter(torch::jit::Value *value) {
  auto *record = _mapper.rawTensorRecord(value);
  ERROR_ON_MSG(record == nullptr,
               "JIT value not tracked " << reinterpret_cast<void *>(value));
  return poptorch::isParameter(*record->tensor_impl);
}

void IDispatch::setParameterName(const at::Tensor &tensor,
                                 const std::string &name) {
  _mapper.setParameterName(tensor, name);
}

std::string IDispatch::getParameterName(torch::jit::Value *value) {
  auto *record = _mapper.rawTensorRecord(value);
  if (record == nullptr) {
    logging::trace("JIT value not tracked {}", reinterpret_cast<void *>(value));
    return "";
  }
  ERROR_ON_MSG(!poptorch::isParameter(*record->tensor_impl),
               "%" << value->debugName() << " is not a Parameter");
  auto it = _mapper.ids_name_map.find(record->ipu_tensor_id);
  if (it == _mapper.ids_name_map.end()) {
    return "";
  }
  return it->second;
}

void IDispatch::replaceValue(torch::jit::Value *v_old,
                             torch::jit::Value *v_new) {
  _mapper.replaceValue(v_old, v_new);
}
} // namespace poptorch
