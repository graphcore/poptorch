// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <popart/builder.hpp>
#include <popart/session.hpp>

#include "popart_compiler/Compiler.hpp"
#include "popart_compiler/CompilerOptions.hpp"
#include "popart_compiler/MultiConvBuilder.hpp"

namespace poptorch {

class Compiler;

namespace detail {

class WeightsIO : public popart::IWeightsIO {
public:
  ~WeightsIO() override = default;
  bool contains(popart::TensorId id) const final;
  popart::MutableVoidData weight(popart::TensorId id) const final;
  void registerParameter(const popart::TensorId &id,
                         const popart::TensorInfo &info);
  void updateData(const std::vector<void *> &host_buffers);

private:
  std::map<popart::TensorId, popart::MutableVoidData> _weights;
  std::vector<popart::TensorId> _weights_order;
};

struct CompilerImpl {
public:
  friend Compiler;

  CompilerImpl() : op_builder(popart::Builder::create()), loss("") {
    ids.emplace_back(""); // None tensor
    active_builder = op_builder.get();
  }
  ~CompilerImpl();

  std::unique_ptr<popart::Builder> op_builder;

  // Op_builder is the top level graph. However to support subgraphs we switch
  // between adding ops from each of these subgraphs. All subgraphs are children
  // of the op_builder top level graph.
  popart::Builder *active_builder;

  std::stack<popart::Builder *> if_true_stack;
  std::stack<popart::Builder *> if_false_stack;

  std::map<popart::TensorId, popart::AnchorReturnType> anchors;

  std::vector<popart::TensorId> ids;

  // Input tensors to the session.
  std::map<popart::TensorId, popart::IArray &> popart_incoming;

  // Output tensors for the session.
  std::map<popart::TensorId, popart::IArray &> popart_outgoing;
  std::map<popart::TensorId, std::vector<void *>> outgoing_duplicates;

  std::list<popart::TensorId> outputs;
  // Flat representation of the output shapes
  std::vector<OutputType> output_types;

  // A list to allocate our buffers in so they get released.
  std::list<std::unique_ptr<popart::IArray>> memory_manager;

  std::unique_ptr<popart::Session> session;

  WeightsIO weights;

  bool is_training;

  // Record the final loss, it is guaranteed by previous passes to be just one
  // loss.
  popart::TensorId loss;

  // List of options which have been explicitely set by the user.
  std::set<std::string> options_set;

  popart::SessionOptions popart_options;

  CompilerOptions options;

  // We add operations using a state based system so the user would set the
  // active IPU and all subsequent operations will be added to that IPU until
  // stopped.
  std::int64_t active_ipu{0};
  std::uint64_t active_stage{0};
  std::int64_t active_phase{0};
  // Keep track of what the maximum phase number used is.
  std::int64_t max_phase{0};

  // Number of ipus used (set by createDevice())
  std::uint64_t num_ipus{0};

  std::unordered_set<std::uint64_t> used_ipus;

  // Map of the pytorch variable update group to the popart weight.
  std::map<std::uint64_t, std::vector<popart::TensorId>> grad_update_groups;

  std::unique_ptr<MultiConvBuilder> multi_conv_builder;

  // General helpers.

  // Inserts memory into the list of tensors being output by the model.
  void addMemoryToOutput(poptorch::TensorId id, void *ptr,
                         std::unique_ptr<popart::IArray> &&memory);

  // Domain helpers
  popart::TensorId reshape(const std::vector<popart::TensorId> &inputs,
                           const std::vector<int64_t> &shape);

  void addOutputTensor(const std::vector<popart::TensorId> &inputs);

  void
  addInputTensorFromParentGraph(const std::vector<popart::TensorId> &inputs);

  popart::TensorId
  addUntypedInputTensor(const std::vector<popart::TensorId> &inputs);

  std::vector<popart::TensorId> customOperation(
      const std::vector<popart::TensorId> &args, const std::string &op,
      const std::string &domain, std::int64_t version, std::int64_t num_outputs,
      const std::shared_ptr<std::vector<PopartAttribute>> &attributes);

  popart::TensorId
  recomputationCheckpoint(const std::vector<popart::TensorId> &input);

  popart::TensorId tensorConstant(const std::vector<popart::TensorId> &inputs,
                                  const PopartConstant &constant);

  poptorch::TensorId
  hostSideTensorConstant(const std::vector<popart::TensorId> &inputs,
                         HostSideConstant constant);

  popart::TensorId addNotInPlace(const std::vector<popart::TensorId> &in);

  // Convert a poptorch tensor id to a popart tensor.
  inline popart::TensorId
  convertPoptorchToPopartTensor(poptorch::TensorId inputs) {
    return ids.at(inputs);
  }

  // Convert a list of poptorch tensors to a list of popart tensors.
  std::vector<popart::TensorId>
  convertPoptorchToPopartTensors(const std::vector<poptorch::TensorId> &inputs);

  popart::TensorId randomNormal(const std::vector<popart::TensorId> &inputs,
                                const std::vector<int64_t> &shape, float mean,
                                float scale, const std::string &dtype);

  popart::TensorId randomUniform(const std::vector<popart::TensorId> &inputs,
                                 const std::vector<int64_t> &shape, float high,
                                 float low, const std::string &dtype);

  popart::TensorId ones(const std::vector<popart::TensorId> &inputs,
                        const std::vector<int64_t> &shape,
                        const std::string &dtype);

  popart::TensorId zeros(const std::vector<popart::TensorId> &inputs,
                         const std::vector<int64_t> &shape,
                         const std::string &dtype);

  popart::TensorId zerosOrOnes(const std::vector<popart::TensorId> &inputs,
                               const std::vector<int64_t> &shape,
                               const std::string &dtype, bool zeros);

  void addMultiConvPart(const std::vector<popart::TensorId> &inputs,
                        const std::vector<int64_t> &dilations,
                        const std::vector<int64_t> &kernel_shape,
                        const std::vector<int64_t> &pads,
                        const std::vector<int64_t> &strides);

  std::vector<popart::TensorId> endMultiConv();

  void optimizerGroup(const std::vector<poptorch::TensorId> &inputs,
                      int64_t group) {
    std::vector<popart::TensorId> ins;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(ins),
                   [&](poptorch::TensorId index) { return ids[index]; });

    grad_update_groups.insert({group, ins});
  }

  std::unique_ptr<popart::Optimizer>
  getOptimizer(const std::vector<Optimizer> &optimizers);

  void updateUseModelConfig();
  std::string checkSystemConfig() const;
  template <typename T, typename U>
  void setOptionIfNotSet(T &option, U value, const std::string &name,
                         const std::string &value_as_string) {
    if (options_set.count(name) && option != static_cast<T>(value)) {
      logging::warn("{} forced by the user from default of {}", name,
                    value_as_string);
    } else {
      logging::debug("{} set to value {}", name, value_as_string);
      option = value;
    }
  }

  template <typename T, typename U>
  void setOptionIfNotSet(T &option, U value, const std::string &name) {
    setOptionIfNotSet(option, value, name, std::to_string(value));
  }

  void
  setExecutionStrategyAttributes(const std::set<popart::TensorId> &tensors);

  const HostSideConstant &getHostSideConstant(poptorch::TensorId id) const;

  bool isHostSideConstant(poptorch::TensorId id) const;

  std::shared_ptr<popart::DeviceInfo> createDevice();
  bool waitIfUnavailable() const;
  void attachToDevice();
  void detachFromDevice();
  bool isAttachedToDevice() const;

  template <typename OptimizerType>
  void updateGroups(OptimizerType *optimizer,
                    const std::vector<Optimizer> &optimizers);
  std::string getPopartIR() const;

private:
  // Constants which are simply returned (possibly as part of a tuple/list) and
  // do not need to be input into Popart
  std::unordered_map<poptorch::TensorId, HostSideConstant> _host_side_constants;
  std::shared_ptr<popart::DeviceInfo> _device;
};

} // namespace detail
} // namespace poptorch
