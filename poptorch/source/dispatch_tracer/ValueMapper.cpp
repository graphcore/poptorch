// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "ValueMapper.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

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

    addTensor(t, record->mlir, record->is_const);
    if (record->jit != nullptr) {
      addTensor(t, record->jit, record->is_const);
    }
    return true;
  }

  return false;
}

// Add a tensor to the IR.
void ValueMapper::addTensor(const at::Tensor &t, poptorch_ir::TensorId id,
                            bool is_const) {
  logging::trace("Adding {} to value mapper, MLIR id: {}",
                 static_cast<void *>(t.unsafeGetTensorImpl()), id);
  // If the tensor is already being tracked then we will update the MLIR
  // value being tracked. Otherwise we insert and add the MLIR value.
  auto itr =
      tensors.insert({t.unsafeGetTensorImpl(), TrackedTensor{t, is_const}})
          .first;
  itr->second.mlir = id;
  itr->second.is_const = is_const;

  // If this map insert fails then we add the storage to the existing list.
  auto pair = storage_map.insert({t.storage().unsafeGetStorageImpl(), {}});
  pair.first->second.push_back(&itr->second);
}

void ValueMapper::addTensor(const at::Tensor &t, torch::jit::Value *val,
                            bool is_const) {
  ERROR_ON_MSG(val == nullptr, "torch::jit::Value* cannot be null");
  logging::trace("Adding {} to value mapper, JIT id: {}",
                 static_cast<void *>(t.unsafeGetTensorImpl()),
                 val->debugName());
  // If the tensor is already being tracked then we will update the JIT
  // value being tracked. Otherwise we insert and add the jit value.
  auto itr =
      tensors.insert({t.unsafeGetTensorImpl(), TrackedTensor{t, is_const}});
  itr.first->second.jit = val;
  itr.first->second.is_const = is_const;

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

c10::optional<bool> ValueMapper::tensorIsConst(const at::Tensor &t) {
  auto itr = tensors.find(t.unsafeGetTensorImpl());

  if (itr == tensors.end()) {
    return c10::nullopt;
  }

  return itr->second.is_const;
}

void ValueMapper::markHalfTensor(const at::Tensor &t) {
  half_tensors.insert(t.unsafeGetTensorImpl());
}

bool ValueMapper::isHalfTensor(const at::Tensor &t) {
  return half_tensors.find(t.unsafeGetTensorImpl()) != std::end(half_tensors);
}

void ValueMapper::addTensorList(const TensorList &list,
                                torch::jit::Value *val) {
  logging::trace("Adding tensor list to value mapper, JIT id: {}",
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

} // namespace poptorch
