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
#include "poptorch/DispatchTracer.hpp"
#include "pytorch_bridge/CompilerTypes.hpp"

namespace poptorch {

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
      const std::hash<const torch::jit::Value *> hash_func;
      size_t hash = 11;
      for (const auto *value : list) {
        const size_t hash_next = hash_func(value);
        hash = hash * 31 + hash_next;
      }
      return hash;
    }
  };

public:
  ValueMapper() = default;

  ValueMapper(ValueMapper &&) noexcept;
  ValueMapper &operator=(ValueMapper &&) noexcept;
  ValueMapper(const ValueMapper &) = delete;
  ValueMapper &operator=(const ValueMapper &) = delete;

  ~ValueMapper();

  // Each tensor we are tracking has a short record containing a pointer to the
  // tensor and its corresponding values in the two IRs.
  struct TrackedTensor {
    explicit TrackedTensor(const at::Tensor &tensor);

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
  };

  TrackedTensor *rawTensorRecord(const at::Tensor &t);

  TrackedTensor *rawTensorRecord(torch::jit::Value *val);

  torch::jit::Value *getValueForTensor(const at::Tensor &t);

  poptorch_ir::TensorId getMLIRForTensor(const at::Tensor &t);

  // There are cases where pytorch creates tensors of the wrong type (for
  // example when copying data to the ipu from an tensor of integers). In this
  // case we want to add the tensor anyway without running the type checks
  void addTensorUnchecked(const at::Tensor &t, torch::jit::Value *val);
  void addTensor(const at::Tensor &t, torch::jit::Value *val);

  void addTensor(const at::Tensor &t, poptorch_ir::TensorId id);

  void addTensorList(const TensorList &list, torch::jit::Value *val);

  void addCopiedTensor(const at::TensorImpl *dest, const at::TensorImpl *src);

  torch::jit::Value *getValueForTensorList(const TensorList &list);

  void setParameterName(const at::Tensor &t, const std::string &name);
  std::string getParameterName(torch::jit::Value *value) const;

  void setParameterPerReplica(const std::string &param_name,
                              const at::Tensor &tensor, int comm_group_type,
                              int shards, int variable_retrieval_mode);
  std::optional<PerReplicaSettings>
  getParameterPerReplica(torch::jit::Value *value) const;

  void replaceValue(torch::jit::Value *v_old, torch::jit::Value *v_new);

  IpuTensorDetails *getTensorDetailsForId(uint64_t id) const;
  IpuTensorDetails *getTensorDetailsForMlirId(poptorch_ir::TensorId id) const;

  // Create an alias from the `src_details` tensor details to the tensor
  // described by `dest_details` and `dest_tensor_id`. The source tensor must
  // already be added to this mapper. The MLIR or JIT value will now map to the
  // same tensor details and PopTorch tensor id as the destination tensor.
  void aliasTensor(const std::shared_ptr<IpuTensorDetails> &dest_details,
                   uint64_t dest_tensor_id,
                   const std::shared_ptr<IpuTensorDetails> &src_details);

  // Creating an alias from src to dest as above, but locating the original
  // tensor's shared pointer and id in this ValueMapper given just the raw
  // pointer to both tensor details.
  void aliasTensor(IpuTensorDetails *dest_details,
                   IpuTensorDetails *src_details);

  bool hasMapping(const at::Tensor &t) const;

private:
  // We map each PyTorch tensor to a record of all the metadata we are tracking
  // about that tensor in the tensor map.
  std::unordered_map<IpuTensorDetails *, TrackedTensor> _tensors;

  // Mapping between parameter / buffer names and tensor IDs
  std::unordered_map<std::string, uint64_t> _name_ids_map;
  std::unordered_map<uint64_t, std::string> _ids_name_map;

  std::unordered_map<uint64_t, PerReplicaSettings> _per_replica_map;

  // We also need to map the values to the mlir so we can query the mlir for a
  // given value.
  std::unordered_map<torch::jit::Value *, TrackedTensor *> _values_map;

  // Map each prim::ListConstruct to a corresponding jit output value.
  std::unordered_map<TensorList, torch::jit::Value *, TensorListHash>
      _tensor_lists;

  // For resolving aliases, it's useful to find a TrackedTensor from its id.
  std::unordered_map<uint64_t, IpuTensorDetails *> _ids_tensors_map;
  std::unordered_map<poptorch_ir::TensorId, IpuTensorDetails *>
      _mlir_id_tensors_map;

  void removeMapperFromDetails();
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_VALUE_MAPPER_HPP_
