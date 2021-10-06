// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_VALUE_MAPPER_HPP_
#define POPTORCH_DISPATCH_VALUE_MAPPER_HPP_

#include <torch/csrc/jit/ir/ir.h>
#include <unordered_map>
#include <vector>

namespace poptorch {
/*
 * The value mapper is the core of the tracer functionality. It provides the
 * system by which we map an incoming at::Tensor onto the compiler IRs. We take
 * a tensor and disambiguate it into a torch::jit::Value or poptorch compiler
 * TensorID corresponding to the values we are tracking for that tensor in the
 * JIT/MLIR graphs respectively.
 */
class ValueMapper {
public:
  // Each tensor we are tracking has a short record containing a pointer to the
  // tensor and its corresponding values in the two IRs.
  struct TrackedTensor {
    explicit TrackedTensor(const at::Tensor &tensor)
        : tensor_impl(tensor.unsafeGetTensorImpl()),
          tracker(tensor.getIntrusivePtr()), jit(nullptr) {}

    // The PyTorch tensor impl. Most of the time it is "the tensor" on the
    // PyTorch side. However the autograd can create non-view aliases so
    // sometimes we have to check if we are an alias with the same stride,
    // shape, and storage offset.
    at::TensorImpl *tensor_impl;

    // Tensors can fall out of scope so we maintain a weak pointer to the
    // tensors we are tracking. This prevents their storage from being
    // deallocated. This wouldn't be an issue as the tensor will then not be
    // rereferenced except PyTorch can allocate a new node's pointer and that
    // can overlap with a tensor we previously saw but unbeknownst to us fell
    // out of scope. This was intended to give us a way to remove invalid
    // pointers, however, it turns out just by tracking them even with a weak
    // reference we stop them from ever completely falling out of scope.
    c10::weak_intrusive_ptr<at::TensorImpl> tracker;

    // The value in JIT IR
    torch::jit::Value *jit;

    // Autograd can create tensors which mirror the original tensor storage.
    bool isSame(const at::Tensor &other) const {
      // If we share storage then we are an alias.
      return tensor_impl->storage().is_alias_of(other.storage());
    }
  };

  // We map each PyTorch tensor to a record of all the metadata we are tracking
  // about that tensor in the tensor map.
  std::unordered_map<at::TensorImpl *, TrackedTensor> tensors;

  // We also map the storage of the tensor to the same structure so we can check
  // for aliases. We do not expect this to be large.
  std::unordered_map<at::StorageImpl *, std::vector<TrackedTensor *>>
      storage_map;

  TrackedTensor *rawTensorRecord(const at::Tensor &t);

  torch::jit::Value *getValueForTensor(const at::Tensor &t);

  void addTensor(const at::Tensor &t, torch::jit::Value *val);

  // Returns true if this is a direct alias and adds it to the approved alias
  // map.
  bool isDirectAlias(const at::Tensor &t);

  // The node we last processed in the graph.
  torch::jit::Node *last_processed_node;
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_VALUE_MAPPER_HPP_
