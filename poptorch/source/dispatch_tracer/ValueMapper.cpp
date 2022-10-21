// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "ValueMapper.hpp"

#include <memory>
#include <utility>

#include "Tensor.hpp"
#include "poptorch/DispatchTracer.hpp"
#include "poptorch/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

ValueMapper::ValueMapper(ValueMapper &&other) noexcept {
  _tensors = std::move(other._tensors);
  for (auto &[_, tracked_tensor] : _tensors) {
    UNUSED(_);
    tracked_tensor.tensor_details->mapper = this;
  }

  _name_ids_map = std::move(other._name_ids_map);
  _ids_name_map = std::move(other._ids_name_map);
  _per_replica_map = std::move(other._per_replica_map);
  _values_map = std::move(other._values_map);
  _tensor_lists = std::move(other._tensor_lists);
  _ids_tensors_map = std::move(other._ids_tensors_map);
  _mlir_id_tensors_map = std::move(other._mlir_id_tensors_map);
}
ValueMapper &ValueMapper::operator=(ValueMapper &&other) noexcept {
  removeMapperFromDetails();

  _tensors = std::move(other._tensors);
  for (auto &[_, tracked_tensor] : _tensors) {
    UNUSED(_);
    tracked_tensor.tensor_details->mapper = this;
  }
  // Ensure that the destructor of other can't remove this mapper
  other._tensors.clear();

  _name_ids_map = std::move(other._name_ids_map);
  _ids_name_map = std::move(other._ids_name_map);
  _per_replica_map = std::move(other._per_replica_map);
  _values_map = std::move(other._values_map);
  _tensor_lists = std::move(other._tensor_lists);
  _ids_tensors_map = std::move(other._ids_tensors_map);
  _mlir_id_tensors_map = std::move(other._mlir_id_tensors_map);

  return *this;
}

ValueMapper::~ValueMapper() { removeMapperFromDetails(); }

ValueMapper::TrackedTensor::TrackedTensor(const at::Tensor &tensor)
    : tensor_details(getTensorDetails(tensor)), jit(nullptr),
      mlir(poptorch_ir::tensor_error_id), ipu_tensor_id(ipuTensorId(tensor)) {}

void ValueMapper::setParameterName(const at::Tensor &t,
                                   const std::string &name) {
  const IpuTensorId id = ipuTensorId(t);
  if (!isParameter(t) && !t.is_floating_point()) {
    logging::warn("Parameter {}: {} was downgraded to constant because PopART "
                  "doesn't support non floating point parameters",
                  name, str(t));
    return;
  }

  ERROR_ON_MSG(!isParameter(t), "Not a parameter or a buffer: " << str(t));
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
  ERROR_ON_MSG(!itr->second->tensor_details->is_parameter,
               "%" << value->debugName() << " is not a Parameter");
  auto it = _ids_name_map.find(itr->second->ipu_tensor_id);
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
  ERROR_ON_MSG(!itr->second->tensor_details->is_parameter,
               "%" << value->debugName() << " is not a Parameter");
  auto it = _per_replica_map.find(itr->second->ipu_tensor_id);
  if (it == _per_replica_map.end()) {
    return std::nullopt;
  }
  return it->second;
}

// Add a tensor to the IR.
void ValueMapper::addTensor(const at::Tensor &t, poptorch_ir::TensorId id) {
  logging::trace("Adding {} to value mapper {}, MLIR id: {}",
                 static_cast<void *>(t.unsafeGetTensorImpl()),
                 static_cast<void *>(this), id);
  // If the tensor is already being tracked then we will update the MLIR
  // value being tracked. Otherwise we insert and add the MLIR value.
  auto new_details = getTensorDetails(t);
  if (new_details->mapper == nullptr) {
    // If this tensor was created by the JIT dispatcher we want the details
    // to point at the JIT's mapper not the MLIR one, so don't overwrite.
    new_details->mapper = this;
  }
  auto itr = _tensors.insert({new_details.get(), TrackedTensor{t}}).first;
  itr->second.mlir = id;
  ERROR_ON(itr->second.tensor_details != new_details);

  _ids_tensors_map.insert({getIpuTensorId(t), new_details.get()});
  _mlir_id_tensors_map.emplace(id, new_details.get());
}

void ValueMapper::addTensorUnchecked(const at::Tensor &t,
                                     torch::jit::Value *val) {
  logging::trace("Adding {} to value mapper {}, JIT ir: {}",
                 static_cast<void *>(t.unsafeGetTensorImpl()),
                 static_cast<void *>(this), val->debugName());

  // If the tensor is already being tracked then we will update the JIT
  // value being tracked. Otherwise we insert and add the jit value.
  auto new_details = getTensorDetails(t);
  new_details->mapper = this;
  auto itr = _tensors.insert({new_details.get(), TrackedTensor{t}}).first;
  itr->second.jit = val;
  ERROR_ON(itr->second.tensor_details != new_details);

  // Ensure we maintain a lookup of torch::jit to pytorch tensor.
  _values_map.insert({val, &itr->second});

  _ids_tensors_map.insert({getIpuTensorId(t), new_details.get()});
}
void ValueMapper::addTensor(const at::Tensor &t, torch::jit::Value *val) {
  ERROR_ON_MSG(val == nullptr, "torch::jit::Value* cannot be null");
  validateTensorShapeAndType(val, t);

  addTensorUnchecked(t, val);
}

void ValueMapper::addCopiedTensor(const at::TensorImpl *dest,
                                  const at::TensorImpl *src) {
  auto src_details = getTensorDetails(*src);
  auto dest_details = getTensorDetails(*dest);
  dest_details->mapper = this;

  aliasTensor(dest_details, ipuTensorId(*dest), src_details);
}

void ValueMapper::aliasTensor(
    const std::shared_ptr<IpuTensorDetails> &dest_details,
    IpuTensorId dest_tensor_id,
    const std::shared_ptr<IpuTensorDetails> &src_details) {
  auto itr = _tensors.find(src_details.get());
  ERROR_ON_MSG(itr == _tensors.end(), "Could not find source tensor");
  auto itr_new =
      _tensors.insert_or_assign(dest_details.get(), itr->second).first;
  itr_new->second.tensor_details = dest_details;
  itr_new->second.ipu_tensor_id = dest_tensor_id;
}

void ValueMapper::aliasTensor(IpuTensorDetails *dest_details,
                              IpuTensorDetails *src_details) {
  auto dest_it = _tensors.find(dest_details);
  auto src_it = _tensors.find(src_details);

  ERROR_ON(dest_it == _tensors.end());
  ERROR_ON(src_it == _tensors.end());

  aliasTensor(dest_it->second.tensor_details, dest_it->second.ipu_tensor_id,
              src_it->second.tensor_details);
}

ValueMapper::TrackedTensor *ValueMapper::rawTensorRecord(const at::Tensor &t) {
  auto itr = _tensors.find(getTensorDetails(t).get());

  if (itr != _tensors.end()) {
    return &itr->second;
  }

  return nullptr;
}

ValueMapper::TrackedTensor *
ValueMapper::rawTensorRecord(torch::jit::Value *val) {
  auto itr = _values_map.find(val);
  if (itr != _values_map.end()) {
    return itr->second;
  }

  return nullptr;
}

// Get the user tensor from our SSA tensors.
torch::jit::Value *ValueMapper::getValueForTensor(const at::Tensor &t) {
  if (!isIpuTensor(t)) {
    return nullptr;
  }
  auto itr = _tensors.find(getTensorDetails(t).get());

  if (itr != _tensors.end()) {
    return itr->second.jit;
  }

  return nullptr;
}

poptorch_ir::TensorId ValueMapper::getMLIRForTensor(const at::Tensor &t) {
  if (!isIpuTensor(t)) {
    return poptorch_ir::tensor_error_id;
  }
  auto itr = _tensors.find(getTensorDetails(t).get());

  if (itr != _tensors.end()) {
    return itr->second.mlir;
  }

  return poptorch_ir::tensor_error_id;
}

bool ValueMapper::hasMapping(const at::Tensor &t) const {
  return _tensors.find(getTensorDetails(t).get()) != _tensors.end();
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

IpuTensorDetails *ValueMapper::getTensorDetailsForId(IpuTensorId id) const {
  auto it = _ids_tensors_map.find(id);
  if (it == _ids_tensors_map.end()) {
    return nullptr;
  }

  return it->second;
}

IpuTensorDetails *
ValueMapper::getTensorDetailsForMlirId(poptorch_ir::TensorId id) const {
  auto it = _mlir_id_tensors_map.find(id);
  if (it == _mlir_id_tensors_map.end()) {
    return nullptr;
  }

  return it->second;
}

void ValueMapper::removeMapperFromDetails() {
  for (auto &[_, tracked_tensor] : _tensors) {
    UNUSED(_);
    tracked_tensor.tensor_details->mapper = nullptr;
  }
}
} // namespace poptorch
