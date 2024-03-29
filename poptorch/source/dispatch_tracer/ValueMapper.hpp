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
    explicit TrackedTensor(const std::shared_ptr<IpuTensorDetails> &details);

    // The underlying tensor information. Note that we don't participate in
    // ownership here. We want to tie the lifetime of the tensor details to the
    // when the tensor is accessible from pytorch. Note that it isn't sufficient
    // to check whether the tensor is directly accessible from pytorch because
    // the tensor details might be kept alive at the end of a chain of view
    // tensors
    std::weak_ptr<IpuTensorDetails> tensor_details;

    bool is_parameter = false;

    // We want to track the lifetime of the tensor_details and the buffer
    // separately. This is so we can get the data from inputs to the graph that
    // are temporaries without extending their lifetime
    std::shared_ptr<Buffer> buffer;

    // The value in JIT IR
    torch::jit::Value *jit = nullptr;

    // The value in our mlir backend.
    poptorch_ir::TensorId mlir = poptorch_ir::tensor_error_id;
  };

  TrackedTensor *rawTensorRecord(const at::Tensor &t);

  TrackedTensor *rawTensorRecord(torch::jit::Value *val);

  torch::jit::Value *getValueForTensor(const at::Tensor &t);

  poptorch_ir::TensorId getMLIRForTensorId(IpuTensorId tensor_id) const;
  poptorch_ir::TensorId getMLIRForTensor(const IpuTensorDetails &details) const;
  poptorch_ir::TensorId getMLIRForTensor(const at::Tensor &t) const;

  void addTensorUnchecked(const at::Tensor &t, torch::jit::Value *val,
                          bool is_param);
  void addTensor(const at::Tensor &t, torch::jit::Value *val, bool is_param);

  void addTensor(const std::shared_ptr<IpuTensorDetails> &details,
                 poptorch_ir::TensorId mlir_id, bool is_param);
  void addTensor(const at::Tensor &t, poptorch_ir::TensorId mlir_id,
                 bool is_param);

  void addTensorList(const TensorList &list, torch::jit::Value *val);

  torch::jit::Value *getValueForTensorList(const TensorList &list);

  bool isParameter(const at::Tensor &t) const;

  void setParameterName(const at::Tensor &t, const std::string &name);
  std::string getParameterName(torch::jit::Value *value) const;

  void setParameterPerReplica(const std::string &param_name,
                              const at::Tensor &tensor, int comm_group_type,
                              int shards, int variable_retrieval_mode);
  std::optional<PerReplicaSettings>
  getParameterPerReplica(torch::jit::Value *value) const;

  void replaceValue(torch::jit::Value *v_old, torch::jit::Value *v_new);

  std::shared_ptr<IpuTensorDetails> getTensorDetailsForId(IpuTensorId id) const;
  std::shared_ptr<IpuTensorDetails>
  getTensorDetailsForMlirId(poptorch_ir::TensorId mlir_id) const;

  Buffer getBufferForId(IpuTensorId id) const;
  poptorch_ir::CpuBuffer getBufferForValue(torch::jit::Value *value) const;

  bool hasMapping(const at::Tensor &t) const;

private:
  // We map each PyTorch tensor to a record of all the metadata we are tracking
  // about that tensor in the tensor map.
  std::unordered_map<IpuTensorId, TrackedTensor> _tensors;

  // Mapping between parameter / buffer names and tensor IDs
  std::unordered_map<std::string, IpuTensorId> _name_ids_map;
  std::unordered_map<IpuTensorId, std::string> _ids_name_map;

  std::unordered_map<IpuTensorId, PerReplicaSettings> _per_replica_map;

  // We also need to map the values to the mlir so we can query the mlir for a
  // given value.
  std::unordered_map<torch::jit::Value *, IpuTensorId> _values_map;

  // Map each prim::ListConstruct to a corresponding jit output value.
  std::unordered_map<TensorList, torch::jit::Value *, TensorListHash>
      _tensor_lists;

  // For resolving aliases, it's useful to find a TrackedTensor from its id.
  std::unordered_map<poptorch_ir::TensorId, IpuTensorId> _mlir_id_tensors_map;

  TrackedTensor *find(const IpuTensorDetails &details);
  const TrackedTensor *find(const IpuTensorDetails &details) const;

  TrackedTensor *find(const at::Tensor &t);
  const TrackedTensor *find(const at::Tensor &t) const;
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_VALUE_MAPPER_HPP_
