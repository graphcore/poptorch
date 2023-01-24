// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_DISPATCH_TRACER_HPP_
#define INCLUDE_POPTORCH_DISPATCH_TRACER_HPP_

#include <cstdint>
#include <functional>
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

namespace poptorch {

struct CompilerOptions;
struct InplaceGraphInfo;
struct PoptorchErrorInfo;

// Toggled by the user in python to choose which backend to target when tracing.
// CPU and SENTINEL will only be toggled by us.
enum TracingMode {
  // Compile normal JIT to run via PopART
  POPART,
};

struct PerReplicaSettings {
  int comm_group_type;
  int shards;
  int variable_retrieval_mode;
  int64_t size0;
  std::shared_ptr<std::vector<char>> host_buffer;
};

// Create a new graph.
void createGraph(TracingMode mode, const std::vector<at::Tensor> &inputs,
                 const CompilerOptions &options);

// The current graph is complete: finalize it.
//
// Trying to add ops after this call is undefined behaviour.
void finalizeGraph();

InplaceGraphInfo getInplaceGraphInfo(size_t num_anchors,
                                     bool replicas_needing_broadcast);

// Get the captured JIT graph. In reality is just returning the
// torch::jit::Graph it's already been compiling during the dispatch process.
std::shared_ptr<torch::jit::Graph> getTracedGraph();

// Get a pointer to the data source for an IPU input / parameter tensor.
// If the value is not a parameter or an input, return nullptr.
void *getDataSource(const at::Tensor &tensor);

void setParameterName(const at::Tensor &tensor, const std::string &name);

// Return the name of a parameter or an empty string if no name was set.
std::string getParameterName(torch::jit::Value *value);

void setParameterPerReplica(const std::string &param_name,
                            const at::Tensor &tensor, int comm_group_type,
                            int shards, int variable_retrieval_mode);

bool getParameterPerReplica(torch::jit::Value *value,
                            PerReplicaSettings &settings);

// Get a pointer to the data source for a given JIT value.
// The value must be an IPU value.
// If the value is not a parameter or an input, return nullptr.
void *getDataSourceForValue(torch::jit::Value *value);

// Return true if the given IPU tensor is a parameter.
bool isParameter(torch::jit::Value *value);

// Start capturing calls.
void startDispatch();

// Stop capturing calls.
void endDispatch(bool error_occurred = false);

// Called before starting to move parameters between the CPU and the IPU.
// (This is used to differentiate inputs from parameters / buffers)
// We expect something like:
// >>> poptorch_core.startParametersMove()
// >>> my_model.to("ipu")
// >>> poptorch_core.endParametersMove()
void startParametersMove();
void endParametersMove();

// Called before starting to move outputs from the IPU to the CPU.
// Allows us to error if an attempt is made to move outputs outside
// of IPUScope.outputs().
void startOutputsMove();
void endOutputsMove();

// Return true if we should be compiling with the dispatcher.
bool isCompilingWithDispatcher();

// Cleans up all objects associated with poptorch
void poptorchAtExit();

// Destroy the active dispatcher object.
void destroyDispatcher();

void replaceValueDispatcher(torch::jit::Value *v_old, torch::jit::Value *v_new);

std::uint64_t getIpuTensorId(const at::Tensor &tensor);

using PoptorchErrorThrower = std::function<void(const PoptorchErrorInfo &info)>;

// Set the function to use to throw python PoptorchError exceptions.
void setPoptorchErrorThrower(PoptorchErrorThrower thrower);

// Throw an exception using the poptorch error thrower.
// Note: used by RegisterAtenOverloads.cpp in a template, that's why it needs
// to be declared publicly.
void throwPoptorchError(const PoptorchErrorInfo &info);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_DISPATCH_TRACER_HPP_
