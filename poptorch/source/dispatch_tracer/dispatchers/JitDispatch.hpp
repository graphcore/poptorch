// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_JIT_DISPATCH_HPP_
#define POPTORCH_DISPATCH_JIT_DISPATCH_HPP_

#include <torch/csrc/jit/ir/ir.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../ValueMapper.hpp"
#include "IDispatch.hpp"

namespace poptorch {

class JITDispatch final : public IDispatch {
public:
  // The JIT graph we are building up.
  torch::jit::Graph graph;

  void createGraph(const std::vector<at::Tensor> &inputs,
                   const std::vector<at::Tensor> &parameters);

  void markOutputs(const std::vector<at::Tensor> &outputs,
                   const std::vector<at::Tensor> &persistent_data_storage,
                   const std::string &output_structure);

  void setCurrentCodeLocation(const torch::jit::SourceRange &source_location);
  void fallback(const c10::OperatorHandle &op, c10::Stack *stack);

  at::Tensor detach(const at::Tensor &self);

  void registerEmptyTensor(const at::Tensor &tensor);

  // We can't control these function signatures.
  // clang-format off
  at::Tensor
  toCopyInplace(const at::Tensor &self,
           c10::optional<at::ScalarType> dtype = c10::nullopt,
           c10::optional<at::Layout> layout = c10::nullopt,
           c10::optional<at::Device> device = c10::nullopt,
           c10::optional<bool> pin = c10::nullopt,
           c10::optional<c10::MemoryFormat> fmt = c10::nullopt);

  const at::Tensor &copyInplace(const at::Tensor &self,
                                const at::Tensor &other);

  // node will be updated to the new target post canonicalisation
  void canonicaliseAndFixOutput(const c10::FunctionSchema &schema,
                              c10::Stack &stack,
                              torch::jit::Node **node,
                              ValueMapper &mapper);

  // clang-format on
private:
  // We use the value mapper to map between incoming at::Tensors and JIR/MLIR
  // types.
  ValueMapper _mapper;
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_JIT_DISPATCH_HPP_
