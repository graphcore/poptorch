// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "ValueMapper.hpp"

#include <memory>
#include <utility>
#include <variant>

#include "Tensor.hpp"
#include "poptorch/DispatchTracer.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"
#include "pytorch_bridge/IpuSession.hpp"

namespace poptorch {

ValueMapper::ValueMapper(ValueMapper &&other) noexcept = default;
ValueMapper &ValueMapper::operator=(ValueMapper &&other) noexcept = default;
ValueMapper::~ValueMapper() = default;

ValueMapper::TrackedTensor::TrackedTensor(
    const std::shared_ptr<IpuTensorDetails> &details)
    : tensor_details(details), buffer(details->getOwningBuffer()) {}

bool ValueMapper::isParameter(const at::Tensor &t) const {
  if (const auto *record = find(t)) {
    return record->is_parameter;
  }
  return false;
}

void ValueMapper::setParameterName(const at::Tensor &t,
                                   const std::string &name) {
  const IpuTensorId id = ipuTensorId(t);
  const auto itr = _tensors.find(id);

  if (itr == _tensors.end()) {
    logging::warn("Parameter {} cannot be named because it was not added to "
                  "the value mapper.",
                  name);
    return;
  }

  if (!itr->second.is_parameter && !t.is_floating_point()) {
    logging::warn("Parameter {}: {} was downgraded to constant because PopART "
                  "doesn't support non floating point parameters",
                  name, str(t));
    return;
  }

  ERROR_ON_MSG(!itr->second.is_parameter,
               "Not a parameter or a buffer: " << str(t));
  auto name_it = _name_ids_map.find(name);
  if (name_it != _name_ids_map.end()) {
    ERROR_ON_MSG(name_it->second != id,
                 "Name " << name << " can't be associated to " << id
                         << " because it is already associated to "
                         << name_it->second);
    return;
  }
  auto id_it = _ids_name_map.find(id);
  if (id_it != _ids_name_map.end()) {
    ERROR_ON_MSG(id_it->second != name, "Name for tensor id "
                                            << id << " can't be set to " << name
                                            << " because it is already set to "
                                            << id_it->second);
    return;
  }

  _name_ids_map.insert({name, id});
  _ids_name_map.insert({id, name});
}
std::string ValueMapper::getParameterName(torch::jit::Value *value) const {
  auto itr = _values_map.find(value);
  if (itr == _values_map.end()) {
    logging::trace("JIT value not tracked {}", reinterpret_cast<void *>(value));
    return "";
  }
  auto it = _ids_name_map.find(itr->second);
  if (it == _ids_name_map.end()) {
    return "";
  }
  return it->second;
}

void ValueMapper::setParameterPerReplica(const std::string &param_name,
                                         const at::Tensor &tensor,
                                         int comm_group_type, int shards,
                                         int variable_retrieval_mode) {
  auto param_it = _name_ids_map.find(param_name);
  if (param_it == std::end(_name_ids_map)) {
    logging::warn("Parameter name {} was not found", param_name);
    return;
  }
  auto data_size = tensorDataSize(tensor);
  ERROR_ON_MSG(!tensor.is_contiguous(),
               "Data source must be contiguous: " << str(tensor));
  const PerReplicaSettings settings = {
      comm_group_type, shards, variable_retrieval_mode, tensor.size(0),
      std::make_shared<std::vector<char>>(data_size)};
  memcpy(settings.host_buffer->data(), tensor.data_ptr(), data_size);
  _per_replica_map[param_it->second] = settings;
}
std::optional<PerReplicaSettings>
ValueMapper::getParameterPerReplica(torch::jit::Value *value) const {
  auto itr = _values_map.find(value);
  if (itr == _values_map.end()) {
    logging::trace("JIT value not tracked {}", reinterpret_cast<void *>(value));
    return std::nullopt;
  }
  auto it = _per_replica_map.find(itr->second);
  if (it == _per_replica_map.end()) {
    return std::nullopt;
  }
  return it->second;
}

// Add a tensor to the IR.
void ValueMapper::addTensor(const std::shared_ptr<IpuTensorDetails> &details,
                            poptorch_ir::TensorId mlir_id, bool is_param) {
  logging::trace("Adding {} to value mapper {}, MLIR id: {}",
                 details->tensor_id, static_cast<void *>(this), mlir_id);

  auto tensor_id = details->tensor_id;
  auto &record =
      _tensors.insert({tensor_id, TrackedTensor{details}}).first->second;
  record.mlir = mlir_id;
  record.is_parameter |= is_param;

  _mlir_id_tensors_map.emplace(mlir_id, tensor_id);
}

void ValueMapper::addTensor(const at::Tensor &t, poptorch_ir::TensorId mlir_id,
                            bool is_param) {
  addTensor(getTensorDetails(t), mlir_id, is_param);
}

void ValueMapper::addTensorUnchecked(const at::Tensor &t,
                                     torch::jit::Value *val, bool is_param) {
  logging::trace("Adding {} to value mapper {}, JIT ir: {}",
                 static_cast<void *>(t.unsafeGetTensorImpl()),
                 static_cast<void *>(this), val->debugName());

  // If the tensor is already being tracked then we will update the JIT
  // value being tracked. Otherwise we insert and add the jit value.
  const auto &new_details = getTensorDetails(t);

  const auto ipu_tensor_id = new_details->tensor_id;
  auto &record = _tensors.insert({ipu_tensor_id, TrackedTensor{new_details}})
                     .first->second;
  record.jit = val;
  record.is_parameter |= is_param;

  // Ensure we maintain a lookup of torch::jit to pytorch tensor.
  _values_map.insert({val, ipu_tensor_id});
}
void ValueMapper::addTensor(const at::Tensor &t, torch::jit::Value *val,
                            bool is_param) {
  ERROR_ON_MSG(val == nullptr, "torch::jit::Value* cannot be null");
  validateTensorShapeAndType(val, t);

  addTensorUnchecked(t, val, is_param);
}

ValueMapper::TrackedTensor *ValueMapper::rawTensorRecord(const at::Tensor &t) {
  return find(t);
}

ValueMapper::TrackedTensor *
ValueMapper::rawTensorRecord(torch::jit::Value *val) {
  auto itr = _values_map.find(val);
  if (itr == _values_map.end()) {
    return nullptr;
  }
  auto tracked_tensor_itr = _tensors.find(itr->second);
  if (tracked_tensor_itr == _tensors.end()) {
    return nullptr;
  }
  return &tracked_tensor_itr->second;
}

// Get the user tensor from our SSA tensors.
torch::jit::Value *ValueMapper::getValueForTensor(const at::Tensor &t) {
  if (!isIpuTensor(t)) {
    return nullptr;
  }

  if (auto *tracked_tensor = find(t)) {
    return tracked_tensor->jit;
  }

  return nullptr;
}

poptorch_ir::TensorId
ValueMapper::getMLIRForTensorId(IpuTensorId tensor_id) const {
  if (const auto itr = _tensors.find(tensor_id); itr != _tensors.end()) {
    return itr->second.mlir;
  }

  return poptorch_ir::tensor_error_id;
}
poptorch_ir::TensorId
ValueMapper::getMLIRForTensor(const IpuTensorDetails &details) const {
  if (const auto *tracked_tensor = find(details)) {
    return tracked_tensor->mlir;
  }

  return poptorch_ir::tensor_error_id;
}

poptorch_ir::TensorId ValueMapper::getMLIRForTensor(const at::Tensor &t) const {
  if (!isIpuTensor(t)) {
    return poptorch_ir::tensor_error_id;
  }

  return getMLIRForTensor(*getTensorDetails(t));
}

bool ValueMapper::hasMapping(const at::Tensor &t) const {
  return find(t) != nullptr;
}

void ValueMapper::addTensorList(const TensorList &list,
                                torch::jit::Value *val) {
  logging::trace("Adding tensor list to value mapper, JIT ir: {}",
                 val->debugName());
  _tensor_lists.insert({list, val});
}

torch::jit::Value *ValueMapper::getValueForTensorList(const TensorList &list) {
  auto itr = _tensor_lists.find(list);
  if (itr != _tensor_lists.end()) {
    return itr->second;
  }
  return nullptr;
}

void ValueMapper::replaceValue(torch::jit::Value *v_old,
                               torch::jit::Value *v_new) {
  for (auto &rec : _tensors) {
    if (rec.second.jit == v_old) {
      rec.second.jit = v_new;
    }
  }
}

std::shared_ptr<IpuTensorDetails>
ValueMapper::getTensorDetailsForId(IpuTensorId id) const {
  auto it = _tensors.find(id);
  if (it == _tensors.end()) {
    return nullptr;
  }

  return it->second.tensor_details.lock();
}

std::shared_ptr<IpuTensorDetails>
ValueMapper::getTensorDetailsForMlirId(poptorch_ir::TensorId id) const {
  auto it = _mlir_id_tensors_map.find(id);
  if (it == _mlir_id_tensors_map.end()) {
    return nullptr;
  }

  return getTensorDetailsForId(it->second);
}

Buffer ValueMapper::getBufferForId(IpuTensorId id) const {
  const auto it = _tensors.find(id);
  if (it == _tensors.end()) {
    return Buffer();
  }

  return *it->second.buffer;
}

poptorch_ir::PopitMemPtr
ValueMapper::getBufferForMlirId(poptorch_ir::TensorId id) const {
  auto it = _mlir_id_tensors_map.find(id);
  if (it == _mlir_id_tensors_map.end()) {
    return nullptr;
  }

  if (auto b = getBufferForId(it->second); b.hasData()) {
    return b.getPopitData();
  }

  return nullptr;
}

poptorch_ir::CpuBuffer
ValueMapper::getBufferForValue(torch::jit::Value *value) const {
  auto itr = _values_map.find(value);
  if (itr == _values_map.end()) {
    return nullptr;
  }

  if (auto b = getBufferForId(itr->second); b.hasData()) {
    return b.getCpuData();
  }

  return nullptr;
}

ValueMapper::TrackedTensor *ValueMapper::find(const IpuTensorDetails &details) {
  auto itr = _tensors.find(details.tensor_id);
  if (itr == _tensors.end()) {
    return nullptr;
  }
  return &itr->second;
}
const ValueMapper::TrackedTensor *
ValueMapper::find(const IpuTensorDetails &details) const {
  return const_cast<ValueMapper *>(this)->find(details);
}

ValueMapper::TrackedTensor *ValueMapper::find(const at::Tensor &t) {
  return find(*getTensorDetails(t));
}
const ValueMapper::TrackedTensor *ValueMapper::find(const at::Tensor &t) const {
  return find(*getTensorDetails(t));
}
} // namespace poptorch
