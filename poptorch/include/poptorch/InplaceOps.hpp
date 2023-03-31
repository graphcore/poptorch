// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_INPLACE_OPS_H
#define INCLUDE_POPTORCH_INPLACE_OPS_H

#include <algorithm>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace c10 {
struct Symbol;
} // namespace c10

namespace torch {
namespace jit {
struct Graph;
struct Node;
using NodeKind = c10::Symbol;
struct Value;
} // namespace jit
} // namespace torch

namespace poptorch {

// Store information related to Graph inputs modified in place.
struct InplaceGraphInfo {
  // Mapping for a graph input which is not modified in place.
  static constexpr size_t no_mapping = std::numeric_limits<std::size_t>::max();

  // Number of outputs from the graph which are not used to emulate
  // inplace ops. (An output may be a list or tuple as well as a tensor).
  size_t num_normal_outputs{0};

  // Number of tensors output from the graph which are not used to
  // emulate inplace ops. (This differs from the previous if the graph returns
  // one or more tuples/lists.)
  size_t num_tensor_outputs{0};

  // Mapping between each input tensor and the output tensor used
  // to update the input. If the input tensor is not changed in place, it will
  // be equal to InplaceGraphInfo::no_mapping
  //
  // Note: these are all Graph inputs (inputs and parameters) but only inputs
  // can have a mapping.
  //
  // If the input at graph_input_idx is modified in place:
  //   m[graph_input_idx] = graph_output_idx
  // else
  //   m[graph_input_idx] = no_mapping
  std::vector<std::size_t> input_output_mapping{};
};

// Get the NodeKind corresponding to the outplace version of the given
// inplace op NodeKind
torch::jit::NodeKind outplaceKind(torch::jit::NodeKind kind);

class InplaceInputsTracker {
public:
  void addTensor(torch::jit::Value *input);
  // Find if the given value is an alias for an input, if so remove the alias
  // from the tracker and return the input it was aliasing. If the given value
  // doesn't alias an input return nullptr.
  torch::jit::Value *eraseCurrentAlias(torch::jit::Value *alias);
  void registerAlias(torch::jit::Value *aliased_input,
                     torch::jit::Value *alias);
  InplaceGraphInfo finalizeGraph(torch::jit::Graph &graph, size_t num_anchors,
                                 bool replicas_needing_broadcast);

private:
  // alias -> aliased
  std::unordered_map<torch::jit::Value *, torch::jit::Value *> _aliases;
};

void fixForLoopInputs(torch::jit::Graph &graph);

void verifyIfElseBlocksOrder(const torch::jit::Graph &graph);
} // namespace poptorch

#endif // INCLUDE_POPTORCH_INPLACE_OPS_H
