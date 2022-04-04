// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_JIT_DISPATCH_HPP_
#define POPTORCH_DISPATCH_JIT_DISPATCH_HPP_

#include <torch/csrc/jit/ir/ir.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../ValueMapper.hpp"
#include "IDispatch.hpp"
#include "MlirDispatch.hpp"

namespace poptorch {

class JITDispatch final : public IDispatch {
public:
  // The JIT graph we are building up.
  torch::jit::Graph graph;

  at::Tensor addInput(const at::Tensor &cpu_tensor) final;
  at::Tensor addParameter(const at::Tensor &cpu_tensor) final;
  void addOutput(const at::Tensor &ipu_src, const at::Tensor &cpu_dest) final;
  void createGraph() final;
  void finalizeGraph() final;

  void
  setCurrentCodeLocation(const torch::jit::SourceRange &source_location) final;
  void fallback(const c10::OperatorHandle &op, c10::Stack *stack);

  at::Tensor detach(const at::Tensor &self) final;

  void registerEmptyTensor(const at::Tensor &tensor) final;

  // We can't control these function signatures.
  at::Tensor
  toCopyInplace(const at::Tensor &self,
                c10::optional<at::ScalarType> dtype = c10::nullopt,
                c10::optional<at::Layout> layout = c10::nullopt,
                c10::optional<at::Device> device = c10::nullopt,
                c10::optional<bool> pin = c10::nullopt,
                c10::optional<c10::MemoryFormat> fmt = c10::nullopt) final;

  const at::Tensor &copyInplace(const at::Tensor &self,
                                const at::Tensor &src) final;

  // Node will be updated to the new target post canonicalisation.
  void canonicaliseAndFixOutput(const c10::FunctionSchema &schema,
                                c10::Stack &stack, torch::jit::Node **node,
                                ValueMapper &mapper);

protected:
  at::Tensor allocateTensorImpl(
      c10::IntArrayRef sizes, c10::optional<at::ScalarType> dtype,
      c10::optional<at::Device> device, c10::optional<at::Layout> layout,
      c10::optional<bool> pin_memory,
      c10::optional<at::MemoryFormat> memory_format) final;

private:
  // We use the value mapper to map between incoming at::Tensors and JIR/MLIR
  // types.
  ValueMapper _mapper;
  uint64_t _next_output_idx{0};
  // We use the MLIR dispatch for shape inference.
  MLIRDispatch _mlir_dispatch;
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_JIT_DISPATCH_HPP_
