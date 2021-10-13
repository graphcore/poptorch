// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_JIT_DISPATCH_HPP_
#define POPTORCH_DISPATCH_JIT_DISPATCH_HPP_

#include <torch/csrc/jit/ir/ir.h>
#include <unordered_map>
#include <vector>

#include "../ValueMapper.hpp"
#include "Tracer.hpp"

namespace poptorch {

class JITDispatch final : public DispatcherBase {
public:
  // The JIT graph we are building up.
  torch::jit::Graph graph;

  void createGraph(const std::vector<at::Tensor> &inputs,
                   const std::vector<at::Tensor> &parameters);

  void markOutputs(const std::vector<at::Tensor> &outputs,
                   const std::vector<at::Tensor> &persistent_data_storage);

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

  at::Tensor &copyInplace(at::Tensor &self, const at::Tensor &other);

  at::Tensor
  convolution(const at::Tensor &input, const at::Tensor &weight,
                    const c10::optional<at::Tensor> &bias,
                    at::IntArrayRef stride, at::IntArrayRef padding,
                    at::IntArrayRef dilation, bool transposed,
                    at::IntArrayRef output_padding, int64_t groups);


  // fake_target will be updated to the new target post canonicalisation
  void canonicaliseAndFixOutput(const c10::FunctionSchema &schema,
                              c10::Stack &stack,
                              torch::jit::Node **fake_target);

  // clang-format on
private:
  // We use the value mapper to map between incoming at::Tensors and JIR/MLIR
  // types.
  ValueMapper _mapper;
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_JIT_DISPATCH_HPP_
