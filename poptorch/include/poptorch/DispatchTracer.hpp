// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_DISPATCH_TRACER_HPP_
#define INCLUDE_POPTORCH_DISPATCH_TRACER_HPP_

#include <memory>
#include <vector>

namespace at {
class Tensor;
}

namespace torch {
namespace jit {
struct Graph;
struct Node;
struct Value;
} // namespace jit
} // namespace torch

namespace poptorch_ir {
class PoptorchExecutorWrapper;
}

namespace poptorch {

namespace detail {
class MLIRExecutableImpl;
}

// Toggled by the user in python to choose which backend to target when tracing.
// CPU and SENTINEL will only be toggled by us.
enum TracingMode {
  // Compile normal JIT to run via PopART
  POPART,
  // Compile via MLIR. Actually uses the JIT path partially under the hood as
  // well.
  MLIR,
  // Run on CPU and print info about operations. Potentially useful to see what
  // gets called and the shapes of outputs ect. TODO(T45468): Renable this.
  CPU,
  // TODO(T45467): We should support running in a sentinel mode where we don't
  // compile anything but can pick up changes. So you would run with MLIR then
  // sentinel on subsequent runs to detect any changes to constants ect in the
  // graph.
  SENTINEL
};

/*
 * When we compile we have two kinds of outputs. JIT or MLIR. JIT just returns
 * the JIT graph to be compiled by a slightly modified compile step in
 * poptorch.cpp. MLIR actually compiles the graph so returns a proper
 * executable which stores all of the state needed to execute the graph.
 */
class MLIRExecutable {
public:
  explicit MLIRExecutable(
      std::unique_ptr<poptorch_ir::PoptorchExecutorWrapper> &&);
  ~MLIRExecutable();
  void execute(const std::vector<at::Tensor> &inputs);
  void weightsToDevice();
  void weightsToHost();

private:
  std::unique_ptr<poptorch_ir::PoptorchExecutorWrapper> _impl;
};

// Create a new graph.
void createGraph(TracingMode mode, const std::vector<at::Tensor> &inputs,
                 const std::vector<at::Tensor> &parameters);

// Mark the outputs of the graph. |outputs| are the outputs as seen in the
// graph, i.e, the tensors as seen/used by the user. `data_storage` are clones
// of them which we should copy the output into. This is to give us a persistent
// return location.
void markOutputs(const std::vector<at::Tensor> &outputs,
                 const std::vector<at::Tensor> &data_storage,
                 bool output_tuple);

// Get the captured JIT graph. In reality is just returning the
// torch::jit::Graph it's already been compiling during the dispatch process.
std::shared_ptr<torch::jit::Graph> getTracedGraph();

// Compile MLIR. Is a full roundtrip compile and spits out a runable poplar
// binary at the end, wrapped by `MLIRExecutable`.
std::shared_ptr<MLIRExecutable> compileMLIR();

// Start capturing calls.
void startDispatch();

// Stop capturing calls.
void endDispatch();

// Return true if the dispatcher is active
bool isDispatcherActive();

// Replace all uses of target with replacement. Keeps track of the replacement
// so it can be correctly handled if the dispatcher is active.
void replaceAllUsesWith(torch::jit::Value *target,
                        torch::jit::Value *replacement);

// Replace all uses of target after node with replacement. Keeps track of the
// replacement so it can be correctly handled if the dispatcher is active.
void replaceAllUsesAfterNodeWith(torch::jit::Node *node,
                                 torch::jit::Value *target,
                                 torch::jit::Value *replacement);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_DISPATCH_TRACER_HPP_
