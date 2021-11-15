// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "ValueMapper.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch {

bool ValueMapper::isDirectAlias(const at::Tensor &t) {
  auto itr = storage_map.find(t.storage().unsafeGetStorageImpl());

  if (itr == storage_map.end()) {
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

    addTensor(t, record->mlir);
    addTensor(t, record->jit);
    return true;
  }

  return false;
}

// Add a tensor to the IR.
void ValueMapper::addTensor(const at::Tensor &t, poptorch_ir::TensorId id) {
  // If the tensor is already being tracked then we will update the MLIR
  // value being tracked. Otherwise we insert and add the MLIR value.
  auto itr = tensors.insert({t.unsafeGetTensorImpl(), TrackedTensor{t}}).first;
  itr->second.mlir = id;

  // If this map insert fails then we add the storage to the existing list.
  auto pair = storage_map.insert({t.storage().unsafeGetStorageImpl(), {}});
  pair.first->second.push_back(&itr->second);
}

void ValueMapper::addTensor(const at::Tensor &t, torch::jit::Value *val) {
  // If the tensor is already being tracked then we will update the JIT
  // value being tracked. Otherwise we insert and add the jit value.
  auto itr = tensors.insert({t.unsafeGetTensorImpl(), TrackedTensor{t}});
  itr.first->second.jit = val;

  // Ensure we maintain a lookup of torch::jit to pytorch tensor.
  values_map.insert({val, &itr.first->second});

  // If this map insert fails then we add the storage to the existing list.
  auto pair = storage_map.insert({t.storage().unsafeGetStorageImpl(), {}});
  pair.first->second.push_back(&itr.first->second);
}

ValueMapper::TrackedTensor *ValueMapper::rawTensorRecord(const at::Tensor &t) {
  auto itr = tensors.find(t.unsafeGetTensorImpl());

  if (itr != tensors.end()) {
    return &itr->second;
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

} // namespace poptorch
