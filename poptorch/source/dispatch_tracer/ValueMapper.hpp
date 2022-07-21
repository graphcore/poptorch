// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_VALUE_MAPPER_HPP_
#define POPTORCH_DISPATCH_VALUE_MAPPER_HPP_

#include <torch/csrc/jit/ir/ir.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Tensor.hpp"
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
    std::shared_ptr<IpuTensorDetails> tensor_details;

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
  std::unordered_map<IpuTensorDetails *, TrackedTensor> tensors;

  // Mapping between parameter / buffer names and tensor IDs
  std::unordered_map<std::string, uint64_t> name_ids_map;
  std::unordered_map<uint64_t, std::string> ids_name_map;

  // We also need to map the values to the mlir so we can query the mlir for a
  // given value.
  std::unordered_map<torch::jit::Value *, TrackedTensor *> values_map;

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

  void addTensorList(const TensorList &list, torch::jit::Value *val);

  void addCopiedTensor(const at::TensorImpl *dest, const at::TensorImpl *src);

  torch::jit::Value *getValueForTensorList(const TensorList &list);

  void setParameterName(const at::Tensor &t, const std::string &name);

  void replaceValue(torch::jit::Value *v_old, torch::jit::Value *v_new);
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_VALUE_MAPPER_HPP_
