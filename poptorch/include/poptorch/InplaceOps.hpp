// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_INPLACE_OPS_H
#define INCLUDE_POPTORCH_INPLACE_OPS_H

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

namespace torch {
namespace jit {
struct Graph;
struct Value;
} // namespace jit
} // namespace torch

namespace poptorch {

// A replacement for and modification to
// torch/csrc/jit/passes/remove_inplace_ops.cpp.
// In addition to replacing inplace ops with their outplace variants,
// the modified algorithm returns as an output any tensors which, in the
// original lowered graph, would correspond to an input tensor modified in
// place. As a result, PopTorch can use returned outputs to update the input
// tensors and, in doing so, emulate inplace op behaviour.
class InplaceOpHandler {
public:
  InplaceOpHandler(const std::shared_ptr<torch::jit::Graph> &graph,
                   size_t num_parameters);

  // Returns the mapping between each input tensor and the output tensor used
  // to update the input. If the input tensor is not changed in place, it will
  // be equal to InplaceOpHandler::no_mapping
  const std::vector<size_t> &getMapping() const {
    return _input_output_mapping;
  }

  // Return the number of outputs from the graph which are not used to emulate
  // inplace ops. (An output may be a list or tuple as well as a tensor).
  size_t getNumNormalOutputs() const { return _num_normal_outputs; }

  // Return the number of tensors output from the graph which are not used to
  // emulate inplace ops. (This differs from the previous if the graph returns
  // one or more tuples/lists.)
  size_t getNumTensorOuputs() const { return _num_normal_tensor_outputs; }

  static constexpr size_t no_mapping = std::numeric_limits<size_t>::max();

private:
  // Store the number of tensors out, including those in tuples.
  void storeNumTensorOutputs();

  // Process the input by changing any nodes to inplace varients and adding an
  // addition output if required.
  void processInput(size_t input);

  torch::jit::Graph *_graph;
  std::vector<torch::jit::Value *> _collapsed_inputs;

  // Map from inputs in, which are modified in place, to "output" which should
  // be used to update the input
  std::vector<size_t> _input_output_mapping;

  // The number of outputs which should be returned in PyTorch. This is the
  // first "num_normal_outputs" in the graph. The rest are used to update inputs
  // which, in the PyTorch model, should be modified in place.
  size_t _num_normal_outputs;

  // Number of tensors: this will differ from the previous in the case of
  // (possibly nested) tuples to include the number of tensors retuurned,
  // including those which are elements of tuples.
  size_t _num_normal_tensor_outputs;
};

} // namespace poptorch

#endif // INCLUDE_POPTORCH_INPLACE_OPS_H
