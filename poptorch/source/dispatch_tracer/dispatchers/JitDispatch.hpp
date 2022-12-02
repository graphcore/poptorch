// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_DISPATCH_JIT_DISPATCH_HPP_
#define POPTORCH_DISPATCH_JIT_DISPATCH_HPP_

#include <torch/csrc/jit/ir/ir.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "pytorch_bridge/CompilerOptions.hpp"

#include "../TypeInferenceHandler.hpp"
#include "../ValueMapper.hpp"
#include "IDispatch.hpp"

#include "poptorch/InplaceOps.hpp"

namespace poptorch {
struct CompilerOptions;

class JITDispatch final : public IDispatch {
public:
  JITDispatch(const CompilerOptions &options, TensorStore *tensor_store);

  // The JIT graph we are building up.
  std::shared_ptr<torch::jit::Graph> graph;

  void addConstant(const at::Tensor &cpu_tensor,
                   const at::Tensor &ipu_tensor) final;
  void addInput(const at::Tensor &cpu_tensor,
                const at::Tensor &ipu_tensor) final;
  void addParameter(const at::Tensor &cpu_tensor,
                    const at::Tensor &ipu_tensor) final;
  void addOutput(const at::Tensor &ipu_src, const at::Tensor &cpu_dest) final;
  void finalizeGraph() final;

  void fallback(const c10::OperatorHandle &op, c10::Stack *stack) override;

  void detach(const c10::OperatorHandle &op, c10::Stack *stack,
              bool moving_parameters) final;

  void registerEmptyTensor(const at::Tensor &tensor, bool is_param) final;

  // Node will be updated to the new target post canonicalisation.
  void fixOutput(c10::Stack &stack, torch::jit::Node *node);

  InplaceGraphInfo finalizeInplaceGraphInfo(size_t num_anchors,
                                            bool replicas_needing_broadcast);

private:
  void addTensor(const at::Tensor &cpu_tensor, const at::Tensor &ipu_tensor,
                 bool is_parameter);

  const std::vector<std::vector<char>> &getSourceLocationExcludes() const final;
  void
  setCurrentCodeLocation(const torch::jit::SourceRange &source_location) final;

  void addTensorToParamNode(const at::Tensor &cpu_tensor);

  CompilerOptions _opts;
  TypeInferenceHandler _type_inference_handler;
  InplaceInputsTracker _inplace_tracker;
};

} // namespace poptorch

#endif // POPTORCH_DISPATCH_JIT_DISPATCH_HPP_
