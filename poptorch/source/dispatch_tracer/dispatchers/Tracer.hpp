// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_TRACER_H_
#define POPTORCH_DISPATCH_TRACER_H_

#include <torch/csrc/jit/ir/ir.h>
#include <unordered_map>
#include <vector>

#include "../ValueMapper.hpp"

namespace poptorch {

class DispatcherBase {
public:
  virtual ~DispatcherBase() {}

  virtual void createGraph(const std::vector<at::Tensor> &inputs,
                           const std::vector<at::Tensor> &parameters) = 0;

  virtual void
  markOutputs(const std::vector<at::Tensor> &ids,
              const std::vector<at::Tensor> &persistent_data_storage) = 0;

  // The "catch-all" fallback kernel.
  virtual void fallback(const c10::OperatorHandle &op, c10::Stack *stack) = 0;

  virtual at::Tensor detach(const at::Tensor &self) = 0;

  // Rather than have each empty overload requring a specialised kernel we
  // simply ask the dispatchers to acknowledge the created empty tensor and we
  // create it manually in the base function registration.
  virtual void registerEmptyTensor(const at::Tensor &empty) = 0;

  virtual at::Tensor
  toCopyInplace(const at::Tensor &self,
                c10::optional<at::ScalarType> dtype = c10::nullopt,
                c10::optional<at::Layout> layout = c10::nullopt,
                c10::optional<at::Device> device = c10::nullopt,
                c10::optional<bool> pin = c10::nullopt,
                c10::optional<c10::MemoryFormat> fmt = c10::nullopt) = 0;

  virtual at::Tensor &copyInplace(at::Tensor &self, const at::Tensor &src) = 0;

  virtual at::Tensor
  convolution(const at::Tensor &input, const at::Tensor &weight,
              const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
              at::IntArrayRef padding, at::IntArrayRef dilation,
              bool transposed, at::IntArrayRef output_padding,
              int64_t groups) = 0;
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_TRACER_H_
