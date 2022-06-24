// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "ValueMapper.hpp"

#include "Tensor.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

bool ValueMapper::TrackedTensor::isSame(const at::Tensor &other) const {
  return ipuTensorId(other) == ipu_tensor_id;
}

ValueMapper::TrackedTensor::TrackedTensor(const at::Tensor &tensor,
                                          bool _is_empty)
    : tensor_impl(tensor.unsafeGetTensorImpl()),
      tracker(tensor.getIntrusivePtr()), jit(nullptr),
      mlir(poptorch_ir::tensor_error_id), ipu_tensor_id(ipuTensorId(tensor)),
      is_empty(_is_empty) {}

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

bool ValueMapper::isDirectAlias(const at::Tensor &t) {
  if (!isIpuTensor(t)) {
    return false;
  }
  auto itr = ipu_ids_map.find(ipuTensorId(t));

  if (itr == ipu_ids_map.end()) {
    return false;
  }

  // We have a set of "approved" aliases. We do not want to get into the
  // habit of checking the storage keys rather than the TensorIds. This can lead
  // to a very messy situation of tracking which tensor is a view of what, etc.
  // So we allow a narrow set of tensors to be tracked via their storage. We do
  // this specifically for autograd tensors which can be a true alias of a
  // tensor by being a direct storage copy and not a view change.
  for (TrackedTensor *record : itr->second) {
    std::int64_t dim = record->tensor_impl->dim();

    // The same number of elements and dimensions.
    if (record->tensor_impl->numel() != t.numel() ||
        record->tensor_impl->dim() != t.dim()) {
      continue;
    }

    // And that all dimensions and strides are exactly the same.
    bool valid = true;
    for (std::int64_t i = 0; i < dim; ++i) {
      if (t.strides()[i] != record->tensor_impl->strides()[i] ||
          t.sizes()[i] != record->tensor_impl->sizes()[i]) {
        valid = false;
        break;
      }
    }

    if (!valid) {
      continue;
    }

    addTensor(t, record->mlir, record->is_empty);
    if (record->jit != nullptr) {
      addTensor(t, record->jit, record->is_empty);
    }
    return true;
  }

  return false;
}

// Add a tensor to the IR.
void ValueMapper::addTensor(const at::Tensor &t, poptorch_ir::TensorId id,
                            bool is_empty) {
  logging::trace("Adding {} to value mapper, MLIR id: {}",
                 static_cast<void *>(t.unsafeGetTensorImpl()), id);
  // If the tensor is already being tracked then we will update the MLIR
  // value being tracked. Otherwise we insert and add the MLIR value.
  auto itr =
      tensors.insert({t.unsafeGetTensorImpl(), TrackedTensor{t, is_empty}})
          .first;
  itr->second.mlir = id;
  itr->second.is_empty = is_empty;

  // If this map insert fails then we add the storage to the existing list.
  auto pair = ipu_ids_map.insert({ipuTensorId(t), {}});
  pair.first->second.push_back(&itr->second);
}

void ValueMapper::addTensor(const at::Tensor &t, torch::jit::Value *val,
                            bool is_empty) {
  ERROR_ON_MSG(val == nullptr, "torch::jit::Value* cannot be null");
  logging::trace("Adding {} to value mapper, JIT ir: {}",
                 static_cast<void *>(t.unsafeGetTensorImpl()),
                 val->debugName());
  // If the tensor is already being tracked then we will update the JIT
  // value being tracked. Otherwise we insert and add the jit value.
  auto itr =
      tensors.insert({t.unsafeGetTensorImpl(), TrackedTensor{t, is_empty}});
  itr.first->second.jit = val;
  itr.first->second.is_empty = is_empty;

  // Ensure we maintain a lookup of torch::jit to pytorch tensor.
  values_map.insert({val, &itr.first->second});

  // If this map insert fails then we add the storage to the existing list.
  auto pair = ipu_ids_map.insert({ipuTensorId(t), {}});
  pair.first->second.push_back(&itr.first->second);
}

ValueMapper::TrackedTensor *ValueMapper::rawTensorRecord(const at::Tensor &t) {
  auto itr = tensors.find(t.unsafeGetTensorImpl());

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
  auto itr = tensors.find(t.unsafeGetTensorImpl());

  if (itr != tensors.end()) {
    return itr->second.jit;
  }

  return nullptr;
}

poptorch_ir::TensorId ValueMapper::getMLIRForTensor(const at::Tensor &t) {
  auto itr = tensors.find(t.unsafeGetTensorImpl());

  if (itr != tensors.end()) {
    return itr->second.mlir;
  }

  return poptorch_ir::tensor_error_id;
}

poptorch_ir::TensorId ValueMapper::getMLIRForJit(torch::jit::Value *val) {
  auto itr = values_map.find(val);

  if (itr != values_map.end()) {
    return itr->second->mlir;
  }

  return poptorch_ir::tensor_error_id;
}

c10::optional<bool> ValueMapper::tensorIsEmpty(const at::Tensor &t) {
  auto itr = tensors.find(t.unsafeGetTensorImpl());

  if (itr == tensors.end()) {
    return c10::nullopt;
  }

  return itr->second.is_empty;
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

} // namespace poptorch
