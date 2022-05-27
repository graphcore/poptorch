// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_VALUE_MAPPER_HPP_
#define POPTORCH_DISPATCH_VALUE_MAPPER_HPP_

#include <torch/csrc/jit/ir/ir.h>

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pytorch_bridge/CompilerTypes.hpp"

namespace poptorch {

// TODO(T61602) Clean up ValueMapper following dispatcher refactoring.
/*
 * The value mapper is the core of the tracer functionality. It provides the
 * system by which we map an incoming at::Tensor onto the compiler IRs. We take
 * a tensor and disambiguate it into a torch::jit::Value or poptorch compiler
 * TensorID corresponding to the values we are tracking for that tensor in the
 * JIT/MLIR graphs respectively.
 */
class ValueMapper {
private:
  using TensorList = std::vector<torch::jit::Value *>;

  // Hash combine for mapping a vector of jit values (inputs of a
  // prim::ListConstruct) to the output jit value. This allows us to use an
  // unordered_map from TensorList to the output values and thus track the
  // incoming tensor lists. Performance and collisions are not very critical
  // in this scenario as we don't expect models with unreasonably
  // large number of lists.
  struct TensorListHash {
    size_t operator()(const TensorList &list) const {
      std::hash<const torch::jit::Value *> hash_func;
      size_t hash = 11;
      for (const auto *value : list) {
        size_t hash_next = hash_func(value);
        hash = hash * 31 + hash_next;
      }
      return hash;
    }
  };

public:
  // Each tensor we are tracking has a short record containing a pointer to the
  // tensor and its corresponding values in the two IRs.
  struct TrackedTensor {
    explicit TrackedTensor(const at::Tensor &tensor, bool _is_empty);

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
    // TODO(T61602) not needed anymore? (We don't use storage as IDs anymore)
    c10::weak_intrusive_ptr<at::TensorImpl> tracker;

    // The value in JIT IR
    torch::jit::Value *jit;

    // The value in our mlir backend.
    poptorch_ir::TensorId mlir;

    // Unique ID of the storage associated to this tensor.
    uint64_t ipu_tensor_id;

    // Is this tensor empty?
    bool is_empty;

    // Autograd can create tensors which mirror the original tensor storage.
    bool isSame(const at::Tensor &other) const;
  };

  // We map each PyTorch tensor to a record of all the metadata we are tracking
  // about that tensor in the tensor map.
  std::unordered_map<at::TensorImpl *, TrackedTensor> tensors;

  // We also map the storage of the tensor to the same structure so we can check
  // for aliases. We do not expect this to be large.
  std::unordered_map<uint64_t, std::vector<TrackedTensor *>> ipu_ids_map;

  // Mapping between parameter / buffer names and tensor IDs
  std::unordered_map<std::string, uint64_t> name_ids_map;
  std::unordered_map<uint64_t, std::string> ids_name_map;

  // We also need to map the values to the mlir so we can query the mlir for a
  // given value.
  std::unordered_map<torch::jit::Value *, TrackedTensor *> values_map;

  // List of tensors which are actually half-valued from our point of view,
  // but which are floats in PyTorch land because the CPU can't handle half
  // typed values.
  std::unordered_set<at::TensorImpl *>
      half_tensors; // TODO(T61602) remove? no boxed CPU execution anymore.

  // Map each prim::ListConstruct to a corresponding jit output value.
  std::unordered_map<TensorList, torch::jit::Value *, TensorListHash>
      tensor_lists;

  TrackedTensor *rawTensorRecord(const at::Tensor &t);

  TrackedTensor *rawTensorRecord(torch::jit::Value *val);

  torch::jit::Value *getValueForTensor(const at::Tensor &t);

  poptorch_ir::TensorId getMLIRForTensor(const at::Tensor &t);

  poptorch_ir::TensorId getMLIRForJit(torch::jit::Value *val);

  c10::optional<bool> tensorIsEmpty(const at::Tensor &t);

  void addTensor(const at::Tensor &t, torch::jit::Value *val,
                 bool is_empty = false);

  void addTensor(const at::Tensor &t, poptorch_ir::TensorId id,
                 bool is_empty = false);

  void markHalfTensor(const at::Tensor &t);

  bool isHalfTensor(const at::Tensor &t);

  void addTensorList(const TensorList &list, torch::jit::Value *val);

  torch::jit::Value *getValueForTensorList(const TensorList &list);

  void setParameterName(const at::Tensor &t, const std::string &name);

  // Returns true if this is a direct alias and adds it to the approved alias
  // map.
  bool isDirectAlias(const at::Tensor &t);

  void replaceValue(torch::jit::Value *v_old, torch::jit::Value *v_new);
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_VALUE_MAPPER_HPP_
