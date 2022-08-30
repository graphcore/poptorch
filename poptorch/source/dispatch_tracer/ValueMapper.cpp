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

ValueMapper::~ValueMapper() {
  for (const auto &entry : tensors) {
    entry.first->mapper = nullptr;
  }
}

ValueMapper::TrackedTensor::TrackedTensor(const at::Tensor &tensor)
    : tensor_details(getTensorDetails(tensor)), jit(nullptr),
      mlir(poptorch_ir::tensor_error_id), ipu_tensor_id(ipuTensorId(tensor)) {}

void ValueMapper::setParameterName(const at::Tensor &t,
                                   const std::string &name) {
  uint64_t id = ipuTensorId(t);
  if (!isParameter(t) && !t.is_floating_point()) {
    logging::warn("Parameter {}: {} was downgraded to constant because PopART "
                  "doesn't support non floating point parameters",
                  name, str(t));
    return;
  }

  ERROR_ON_MSG(!isParameter(t), "Not a parameter or a buffer: " << str(t));
  auto name_it = name_ids_map.find(name);
  if (name_it != name_ids_map.end()) {
    ERROR_ON_MSG(name_it->second != id,
                 "Name " << name << " can't be associated to " << id
                         << " because it is already associated to "
                         << name_it->second);
    return;
  }
  auto id_it = ids_name_map.find(id);
  if (id_it != ids_name_map.end()) {
    ERROR_ON_MSG(id_it->second != name, "Name for tensor id "
                                            << id << " can't be set to " << name
                                            << " because it is already set to "
                                            << id_it->second);
    return;
  }

  name_ids_map.insert({name, id});
  ids_name_map.insert({id, name});
}

// Add a tensor to the IR.
void ValueMapper::addTensor(const at::Tensor &t, poptorch_ir::TensorId id) {
  logging::trace("Adding {} to value mapper {}, MLIR id: {}",
                 static_cast<void *>(t.unsafeGetTensorImpl()),
                 static_cast<void *>(this), id);
  // If the tensor is already being tracked then we will update the MLIR
  // value being tracked. Otherwise we insert and add the MLIR value.
  auto new_details = getTensorDetails(*t.unsafeGetTensorImpl());
  if (new_details->mapper == nullptr) {
    // If this tensor was created by the JIT dispatcher we want the details
    // to point at the JIT's mapper not the MLIR one, so don't overwrite.
    new_details->mapper = this;
  }
  auto itr = tensors.insert({new_details.get(), TrackedTensor{t}}).first;
  itr->second.mlir = id;
  ERROR_ON(itr->second.tensor_details != new_details);

  _ids_tensors_map.insert({getIpuTensorId(t), new_details.get()});
}

void ValueMapper::addTensor(const at::Tensor &t, torch::jit::Value *val) {
  ERROR_ON_MSG(val == nullptr, "torch::jit::Value* cannot be null");
  logging::trace("Adding {} to value mapper {}, JIT ir: {}",
                 static_cast<void *>(t.unsafeGetTensorImpl()),
                 static_cast<void *>(this), val->debugName());
  validateTensorShapeAndType(val, t);

  // If the tensor is already being tracked then we will update the JIT
  // value being tracked. Otherwise we insert and add the jit value.
  auto new_details = getTensorDetails(*t.unsafeGetTensorImpl());
  new_details->mapper = this;
  auto itr = tensors.insert({new_details.get(), TrackedTensor{t}});
  itr.first->second.jit = val;
  ERROR_ON(itr.first->second.tensor_details != new_details);

  // Ensure we maintain a lookup of torch::jit to pytorch tensor.
  values_map.insert({val, &itr.first->second});

  _ids_tensors_map.insert({getIpuTensorId(t), new_details.get()});
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
    uint64_t dest_tensor_id,
    const std::shared_ptr<IpuTensorDetails> &src_details) {
  auto itr = tensors.find(src_details.get());
  ERROR_ON_MSG(itr == tensors.end(), "Could not find source tensor");
  auto itr_new = tensors.insert_or_assign(dest_details.get(), itr->second);
  itr_new.first->second.tensor_details = dest_details;
  itr_new.first->second.ipu_tensor_id = dest_tensor_id;
}

void ValueMapper::aliasTensor(IpuTensorDetails *dest_details,
                              IpuTensorDetails *src_details) {
  auto dest_it = tensors.find(dest_details);
  auto src_it = tensors.find(src_details);

  ERROR_ON(dest_it == tensors.end());
  ERROR_ON(src_it == tensors.end());

  aliasTensor(dest_it->second.tensor_details, dest_it->second.ipu_tensor_id,
              src_it->second.tensor_details);
}

ValueMapper::TrackedTensor *ValueMapper::rawTensorRecord(const at::Tensor &t) {
  auto itr = tensors.find(getTensorDetails(*t.unsafeGetTensorImpl()).get());

  if (itr != tensors.end()) {
    return &itr->second;
  }

  return nullptr;
}

ValueMapper::TrackedTensor *
ValueMapper::rawTensorRecord(torch::jit::Value *val) {
  auto itr = values_map.find(val);
  if (itr != values_map.end()) {
    return itr->second;
  }

  return nullptr;
}

// Get the user tensor from our SSA tensors.
torch::jit::Value *ValueMapper::getValueForTensor(const at::Tensor &t) {
  if (!isIpuTensor(t)) {
    return nullptr;
  }
  auto itr = tensors.find(getTensorDetails(*t.unsafeGetTensorImpl()).get());

  if (itr != tensors.end()) {
    return itr->second.jit;
  }

  return nullptr;
}

poptorch_ir::TensorId ValueMapper::getMLIRForTensor(const at::Tensor &t) {
  if (!isIpuTensor(t)) {
    return poptorch_ir::tensor_error_id;
  }
  auto itr = tensors.find(getTensorDetails(*t.unsafeGetTensorImpl()).get());

  if (itr != tensors.end()) {
    return itr->second.mlir;
  }

  return poptorch_ir::tensor_error_id;
}

void ValueMapper::addTensorList(const TensorList &list,
                                torch::jit::Value *val) {
  logging::trace("Adding tensor list to value mapper, JIT ir: {}",
                 val->debugName());
  tensor_lists.insert({list, val});
}

torch::jit::Value *ValueMapper::getValueForTensorList(const TensorList &list) {
  auto itr = tensor_lists.find(list);
  if (itr != tensor_lists.end()) {
    return itr->second;
  }
  return nullptr;
}

void ValueMapper::replaceValue(torch::jit::Value *v_old,
                               torch::jit::Value *v_new) {
  for (auto &rec : tensors) {
    if (rec.second.jit == v_old) {
      rec.second.jit = v_new;
    }
  }
}

IpuTensorDetails *ValueMapper::getTensorDetailsForId(uint64_t id) const {
  auto it = _ids_tensors_map.find(id);
  if (it == _ids_tensors_map.end()) {
    return nullptr;
  }

  return it->second;
}

} // namespace poptorch
