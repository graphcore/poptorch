// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_DISPATCH_TRACER_HPP_
#define INCLUDE_POPTORCH_DISPATCH_TRACER_HPP_

#include <memory>
#include <string>
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
class PoplarExecutorWrapper;
}

namespace poptorch {

struct InplaceGraphInfo;

// Toggled by the user in python to choose which backend to target when tracing.
// CPU and SENTINEL will only be toggled by us.
enum TracingMode {
  // Compile normal JIT to run via PopART
  POPART,
  // Compile via MLIR. Actually uses the JIT path partially under the hood as
  // well.
  MLIR,
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
 * executor which stores all of the state needed to execute the graph.
 */
class MLIRExecutor {
public:
  explicit MLIRExecutor(std::unique_ptr<poptorch_ir::PoplarExecutorWrapper> &&);
  ~MLIRExecutor();
  void execute(const std::vector<at::Tensor> &inputs);
  void weightsToDevice();
  void weightsToHost();

private:
  std::unique_ptr<poptorch_ir::PoplarExecutorWrapper> _impl;
};

// Create a new graph.
void createGraph(TracingMode mode, const std::vector<at::Tensor> &inputs,
                 const std::vector<std::string> &source_location_excludes);

// The current graph is complete: finalize it.
//
// Trying to add ops after this call is undefined behaviour.
void finalizeGraph();

InplaceGraphInfo getInplaceGraphInfo(size_t num_anchors,
                                     bool replicas_needing_broadcast);

// Get the captured JIT graph. In reality is just returning the
// torch::jit::Graph it's already been compiling during the dispatch process.
std::shared_ptr<torch::jit::Graph> getTracedGraph();

// Compile MLIR. Is a full roundtrip compile and spits out a runable poplar
// binary at the end, wrapped by `MLIRExecutor`.
std::shared_ptr<MLIRExecutor> compileMLIR();

// Get a pointer to the data source for an IPU input / parameter tensor.
// If the value is not a parameter or an input, return nullptr.
void *getDataSource(const at::Tensor &tensor);

void setParameterName(const at::Tensor &tensor, const std::string &name);

// Return the name of a parameter or an empty string if no name was set.
std::string getParameterName(torch::jit::Value *value);

// Get a pointer to the data source for a given JIT value.
// The value must be an IPU value.
// If the value is not a parameter or an input, return nullptr.
void *getDataSourceForValue(torch::jit::Value *value);

// Return true if the given IPU tensor is a parameter.
bool isParameter(torch::jit::Value *value);

// Switch to the eager mode dispatcher.
void enableEagerMode();

// Start capturing calls.
// TODO(T61528): not needed anymore?
void startDispatch();

// Stop capturing calls.
// TODO(T61528): not needed anymore?
void endDispatch(bool error_occurred = false);

// Called before starting to move parameters between the CPU and the IPU.
// (This is used to differentiate inputs from parameters / buffers)
// We expect something like:
// >>> poptorch_core.startParametersMove()
// >>> my_model.to("ipu")
// >>> poptorch_core.endParametersMove()
// TODO(T61576) Find a better way to identify parameters and buffers.
void startParametersMove();
void endParametersMove();

// Return true if we should be compilin with the dispatcher.
bool isCompilingWithDispatcher();

// Destroy the active dispatcher object.
void destroyDispatcher();

// Replace all uses of target with replacement. Keeps track of the replacement
// so it can be correctly handled if the dispatcher is active.
void replaceAllUsesWith(torch::jit::Value *target,
                        torch::jit::Value *replacement);

// Replace all uses of target after node with replacement. Keeps track of the
// replacement so it can be correctly handled if the dispatcher is active.
void replaceAllUsesAfterNodeWith(torch::jit::Node *node,
                                 torch::jit::Value *target,
                                 torch::jit::Value *replacement);

void replaceValueDispatcher(torch::jit::Value *v_old, torch::jit::Value *v_new);
} // namespace poptorch

#endif // INCLUDE_POPTORCH_DISPATCH_TRACER_HPP_
