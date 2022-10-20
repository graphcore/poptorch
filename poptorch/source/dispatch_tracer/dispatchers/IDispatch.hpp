// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_IDISPATCH_H_
#define POPTORCH_IDISPATCH_H_

#include <ATen/Tensor.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/ir.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../ValueMapper.hpp"

namespace poptorch {

class IDispatch {
public:
  explicit IDispatch(TensorStore *tensor_store);

  virtual ~IDispatch();

  // Input tensor is a CPU tensor, returns an IPU tensor.
  virtual void addInput(const at::Tensor &cpu_tensor,
                        const at::Tensor &ipu_tensor) = 0;
  // Constant tensor is a CPU tensor, returns an IPU tensor.
  virtual void addConstant(const at::Tensor &cpu_tensor,
                           const at::Tensor &ipu_tensor) = 0;
  // Input tensor is a CPU tensor, returns an IPU tensor.
  virtual void addParameter(const at::Tensor &cpu_tensor,
                            const at::Tensor &ipu_tensor) = 0;
  // Source tensor is an IPU tensor, destination is a CPU tensor.
  virtual void addOutput(const at::Tensor &ipu_src,
                         const at::Tensor &cpu_dest) = 0;
  virtual void finalizeGraph() = 0;
  // When running in eager mode with use lazy tensor on stop tracing and execute
  // asynchronously
  virtual void markStep() {}

  void setPythonStack(const std::vector<torch::jit::StackEntry> &stack);

  // The "catch-all" fallback kernel.
  virtual void fallback(const c10::OperatorHandle &op, c10::Stack *stack) = 0;

  virtual void detach(const c10::OperatorHandle &op, c10::Stack *stack,
                      bool moving_parameters) = 0;

  // Rather than have each empty overload requring a specialised kernel we
  // simply ask the dispatchers to acknowledge the created empty tensor and we
  // create it manually in the base function registration.
  virtual void registerEmptyTensor(const at::Tensor &empty) = 0;

  void *getDataSource(torch::jit::Value *val);
  bool isParameter(torch::jit::Value *val);

  void replaceValue(torch::jit::Value *v_old, torch::jit::Value *v_new);

  void setParameterName(const at::Tensor &tensor, const std::string &name);
  std::string getParameterName(torch::jit::Value *val);

  void setParameterPerReplica(const std::string &param_name,
                              const at::Tensor &tensor, int comm_group_type,
                              int shards, int variable_retrieval_mode);
  bool getParameterPerReplica(torch::jit::Value *value,
                              PerReplicaSettings &settings);

protected:
  // We use the value mapper to map between incoming at::Tensors and JIR/MLIR
  // types.
  ValueMapper _mapper;

  // Used to create and manage tensors. This is a raw pointer to ensure this is
  // trivially copyable, but must never be nullptr.
  TensorStore *_tensor_store;

  virtual const std::vector<std::vector<char>> &
  getSourceLocationExcludes() const = 0;
  virtual void
  setCurrentCodeLocation(const torch::jit::SourceRange &source_location) = 0;

private:
  torch::jit::SourceRange getPythonInterpreterSourceRange(
      const std::vector<torch::jit::StackEntry> &cs) const;
};

} // namespace poptorch

#endif // POPTORCH_IDISPATCH_H_
