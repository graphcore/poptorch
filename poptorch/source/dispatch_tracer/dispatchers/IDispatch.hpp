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
  virtual ~IDispatch();

  virtual at::Tensor
  allocateTensor(c10::IntArrayRef sizes,
                 c10::optional<at::ScalarType> dtype = c10::nullopt,
                 c10::optional<at::Device> device = c10::nullopt,
                 c10::optional<at::Layout> layout = c10::nullopt,
                 c10::optional<bool> pin_memory = c10::nullopt,
                 c10::optional<at::MemoryFormat> memory_format = c10::nullopt);

  // Input tensor is a CPU tensor, returns an IPU tensor.
  virtual at::Tensor addInput(const at::Tensor &cpu_tensor) = 0;
  // Constant tensor is a CPU tensor, returns an IPU tensor.
  virtual at::Tensor addConstant(const at::Tensor &cpu_tensor) = 0;
  // Input tensor is a CPU tensor, returns an IPU tensor.
  virtual at::Tensor addParameter(const at::Tensor &cpu_tensor) = 0;
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

  // Sets up a an inplace copy in the graph from src to self. Returns self
  // unaltered as a convenience.
  virtual const at::Tensor &copyInplace(const at::Tensor &self,
                                        const at::Tensor &src) = 0;

  void *getDataSource(torch::jit::Value *val);
  bool isParameter(torch::jit::Value *val);

  void replaceValue(torch::jit::Value *v_old, torch::jit::Value *v_new);

  void setParameterName(const at::Tensor &tensor, const std::string &name);
  std::string getParameterName(torch::jit::Value *val);

protected:
  // We use the value mapper to map between incoming at::Tensors and JIR/MLIR
  // types.
  ValueMapper _mapper;

  virtual const std::vector<std::vector<char>> &
  getSourceLocationExcludes() const = 0;
  virtual void
  setCurrentCodeLocation(const torch::jit::SourceRange &source_location) = 0;

private:
  torch::jit::SourceRange getPythonInterpreterSourceRange(
      const std::vector<torch::jit::StackEntry> &cs) const;

  uint64_t _next_tensor_id{1};
};

} // namespace poptorch

#endif // POPTORCH_IDISPATCH_H_
