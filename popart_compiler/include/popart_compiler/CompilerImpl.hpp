// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#pragma once

#include <algorithm>
#include <atomic>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <vector>

#include <popart/builder.hpp>
#include <popart/iarray.hpp>
#include <popart/istepio.hpp>
#include <popart/session.hpp>
#include <popart/stepio.hpp>
#include <popart/voiddata.hpp>
#include <poplar/Tensor.hpp>

#include "popart_compiler/Compiler.hpp"
#include "popart_compiler/CompilerOptions.hpp"
#include "popart_compiler/MultiConvBuilder.hpp"

namespace poptorch {

class Compiler;

namespace detail {

/*
  We use this structure to maintain all the information related to a CPU
  callback. This is used by the custom op to create the poplar tensors and by
  the compiler to create the poplar callbacks.
*/
struct CallbackInternalMetadata {
  // We need a unique ID for each so we can track how many we've added.
  static std::uint32_t number_of_added_ops;

  // The thing we are calling back.
  std::function<void()> the_callback;

  // Pointers to the buffers on host.
  std::vector<void *> input_pointers;
  std::vector<void *> output_pointers;

  // The names of the operation which we give on creation. The custom op needs
  // to see these to create the operation and the compiler needs it to attach
  // the callbacks.
  std::string handle;

  // Type and shape info for the input and outputs.
  std::vector<poptorch::PopartType> input_types;
  std::vector<std::vector<std::size_t>> input_shapes;
  std::vector<poptorch::PopartType> output_types;
  std::vector<std::vector<std::size_t>> output_shapes;

  // The callbacks are called in random order so we need to track how many have
  // copied their data to make sure we only call the host function once all of
  // them have copied it.
  std::atomic<std::uint32_t> number_of_input_streams_inited;
};

class StepIO : public popart::IStepIO {
public:
  struct ArrayInfo {
    popart::IArray &array;
    int64_t offset;
  };

  using ArrayType = popart::IArray;
  using AccessorType = popart::StepIONS::IArrayAccessor;
  using TensorArrayMap = std::map<popart::TensorId, ArrayType &>;
  using TensorTimestamps = std::map<popart::TensorId, std::vector<double>>;
  using TensorArrayInfo = std::map<popart::TensorId, ArrayInfo>;
  using TensorStepDataInfo = std::map<popart::TensorId, popart::TensorInfo>;

  StepIO() = default;

  popart::ConstVoidData in(popart::TensorId id, int64_t num_elems, // NOLINT
                           bool prefetch) override;
  void inComplete(popart::TensorId id, int64_t num_elems) override; // NOLINT
  popart::MutableVoidData out(popart::TensorId id,
                              int64_t num_elems) override; // NOLINT
  void outComplete(popart::TensorId id) override;          // NOLINT

  void computeStepDataInfo(const popart::TensorId &id, popart::IArray *array);
  void populate(const TensorArrayMap &inputs, const TensorArrayMap &outputs);

  template <typename T>
  T get(const popart::TensorId &id, TensorArrayInfo *map, int64_t num_elems);
  static void timestamp(TensorTimestamps *time, const popart::TensorId &id);

  void assertNumElements(
      const popart::popx::Executablex & /*unused*/) const override {}

  const std::vector<double> &
  getInputTimestamps(const popart::TensorId &id) const {
    return in_times.at(id);
  }

  const std::vector<double> &
  getInputCompleteTimestamps(const popart::TensorId &id) const {
    return in_complete_times.at(id);
  }

  const std::vector<double> &
  getOutputTimestamps(const popart::TensorId &id) const {
    return out_times.at(id);
  }

  const std::vector<double> &
  getOutputCompleteTimestamps(const popart::TensorId &id) const {
    return out_complete_times.at(id);
  }

protected:
  TensorArrayInfo inputs_info;
  TensorArrayInfo outputs_info;
  TensorStepDataInfo step_data_info;

  TensorTimestamps in_times;
  TensorTimestamps in_complete_times;
  TensorTimestamps out_times;
  TensorTimestamps out_complete_times;
};

class WeightsIO : public popart::IWeightsIO {
public:
  ~WeightsIO() override = default;
  bool contains(popart::TensorId id) const final;
  popart::MutableVoidData weight(popart::TensorId id) const final;
  void registerParameter(const popart::TensorId &id,
                         const popart::TensorInfo &info);
  void updateData(const std::vector<void *> &host_buffers);
  const std::vector<popart::TensorId> &parameterIds() const;

private:
  std::map<popart::TensorId, popart::MutableVoidData> _weights;
  std::vector<popart::TensorId> _weights_order;
};

struct CompilerImpl {
public:
  friend Compiler;

  CompilerImpl() : op_builder(popart::Builder::create()) {
    ids.emplace_back(""); // None tensor
    ids_types.push_back(PopartType::UNDEFINED);
    active_builder = op_builder.get();
    using_overlapped_io = false;
  }
  ~CompilerImpl();

  std::unique_ptr<popart::Builder> op_builder;

  // Op_builder is the top level graph. However to support subgraphs we switch
  // between adding ops from each of these subgraphs. All subgraphs are children
  // of the op_builder top level graph.
  popart::Builder *active_builder;

  std::map<popart::TensorId, popart::AnchorReturnType> anchors;

  std::vector<popart::TensorId> ids;
  std::vector<PopartType> ids_types;

  // Input tensors to the session.
  std::map<popart::TensorId, popart::IArray &> popart_incoming;

  // Output tensors for the session.
  std::map<popart::TensorId, popart::IArray &> popart_outgoing;
  std::map<popart::TensorId, std::vector<void *>> outgoing_duplicates;

  std::vector<popart::TensorId> inputs;
  std::vector<popart::TensorId> outputs;
  // Flat representation of the output shapes
  std::vector<OutputType> output_types;

  // A list to allocate our buffers in so they get released.
  std::list<std::unique_ptr<popart::IArray>> memory_manager;

  std::unique_ptr<popart::Session> session;

  StepIO stepio;
  WeightsIO weights;
  WeightsIO optim_state_tensors;

  bool is_training;

  // At least one use of overlapped host IO
  bool using_overlapped_io;

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
  // By default, the active IPU is 0 in case setActiveIpu is never used.
  // However, clearActiveIpu will set it to -1 making future use of
  // setActiveIpu compulsory.
  std::int64_t active_ipu{0};
  std::uint64_t active_stage{0};
  std::int64_t active_phase{0};
  // Keep track of what the maximum phase number used is.
  std::int64_t max_phase{0};

  // Number of ipus used (set by createDevice())
  std::uint64_t num_ipus{0};

  // Which IPUs are being used
  // Note that this does not take into account replication and so the number of
  // IPUs actually used is multiplied by popart_options.replicatedGraphCount.
  // Due to rounding and the issues with skipping an IPU in a range, the number
  // of IPUs required may increase further.
  std::unordered_set<std::uint64_t> used_ipus;

  // Keep the number of ipu switches to work out the number of pipeline stages
  // if relevant.
  std::uint64_t num_ipu_switches{0};

  // Store the last ipu used: this will always match active_ipu unless
  // active_ipu is set to -1.
  std::uint64_t last_ipu_used{0};

  // Map of the pytorch variable update group to the popart weight.
  std::map<std::uint64_t, std::vector<popart::TensorId>> grad_update_groups;

  std::unique_ptr<MultiConvBuilder> multi_conv_builder;

  // Dynamic container for all the callbacks to live in.
  std::list<CallbackInternalMetadata> callbacks;

  // Returns the number of pipeline stages in the model execution
  std::uint64_t numPipelineStages();

  // General helpers.

  // Inserts memory into the list of tensors being output by the model.
  void addMemoryToOutput(poptorch::TensorId id, void *ptr,
                         std::unique_ptr<popart::IArray> &&memory);

  // Domain helpers
  popart::TensorId reshape(const std::vector<popart::TensorId> &tensors,
                           const std::vector<int64_t> &shape);

  void addOutputTensor(const std::vector<popart::TensorId> &tensors);

  popart::TensorId
  addUntypedInputTensor(const std::vector<popart::TensorId> &tensors);

  std::vector<popart::TensorId> customOperation(
      const std::vector<popart::TensorId> &args, const std::string &op,
      const std::string &domain, std::int64_t version, std::int64_t num_outputs,
      const std::shared_ptr<std::vector<PopartAttribute>> &attributes);

  popart::TensorId
  recomputationCheckpoint(const std::vector<popart::TensorId> &tensors);

  popart::TensorId tensorConstant(const std::vector<popart::TensorId> &tensors,
                                  const PopartConstant &constant);

  poptorch::TensorId
  hostSideTensorConstant(const std::vector<popart::TensorId> &tensors,
                         HostSideConstant constant);

  popart::TensorId addNotInPlace(const std::vector<popart::TensorId> &in);

  popart::TensorId randomNormal(const std::vector<popart::TensorId> &tensors,
                                const std::vector<int64_t> &shape, float mean,
                                float scale, const std::string &dtype);

  popart::TensorId randomUniform(const std::vector<popart::TensorId> &tensors,
                                 const std::vector<int64_t> &shape, float high,
                                 float low, const std::string &dtype);

  popart::TensorId ones(const std::vector<popart::TensorId> &tensors,
                        const std::vector<int64_t> &shape,
                        const std::string &dtype);

  popart::TensorId zeros(const std::vector<popart::TensorId> &tensors,
                         const std::vector<int64_t> &shape,
                         const std::string &dtype);

  popart::TensorId zerosOrOnes(const std::vector<popart::TensorId> &tensors,
                               const std::vector<int64_t> &shape,
                               const std::string &dtype, bool zeros);

  void addMultiConvPart(const std::vector<popart::TensorId> &tensors,
                        const std::vector<int64_t> &dilations,
                        const std::vector<int64_t> &kernel_shape,
                        const std::vector<int64_t> &pads,
                        const std::vector<int64_t> &strides);

  std::vector<popart::TensorId> endMultiConv();

  void optimizerGroup(const std::vector<poptorch::TensorId> &tensors,
                      int64_t group) {
    std::vector<popart::TensorId> ins;
    std::transform(tensors.begin(), tensors.end(), std::back_inserter(ins),
                   [&](poptorch::TensorId index) { return ids[index]; });

    grad_update_groups.insert({group, ins});
  }

  std::unique_ptr<popart::Optimizer>
  getPopartOptimizer(std::vector<Optimizer> optimizers);

  void updateUseModelConfig();
  std::string checkSystemConfig() const;
  template <typename T, typename U>
  void setOptionIfNotSet(T &option, U value, const std::string &name,
                         const std::string &value_as_string) {
    if (options_set.count(name) && option != static_cast<T>(value)) {
      logging::warn("{} forced by the user from default to {}, "
                    "ignoring value {}",
                    name, option, value_as_string);
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

  // Returns the PopART type for specified id
  PopartType getPopartType(poptorch::TensorId id) const;

  // Caches all PopART types
  void cachePopartTypes();

  // Returns cached PopART type for the specified id
  // Caution: no bounds checking as this is called for each input, each run.
  // cachePopartType must be called once first.
  PopartType getCachedPopartType(poptorch::TensorId id) const {
    return ids_types[id];
  }

  void setAttribute(const std::string &attribute, const std::string &key,
                    const std::string &value);
  void clearAttribute(const std::string &attribute, const std::string &key);

private:
  // Raise an error if cycle logging is enabled
  void errorOnCycleLogging() const;

  // Constants which are simply returned (possibly as part of a tuple/list) and
  // do not need to be input into Popart
  std::unordered_map<poptorch::TensorId, HostSideConstant> _host_side_constants;
  std::shared_ptr<popart::DeviceInfo> _device;

  std::unordered_map<std::string, std::map<std::string, std::string>>
      _attribute_key_value_map;
};

} // namespace detail

popart::DataType popartTypeFromPoptorch(PopartType);

poplar::Type poplarTypeFromPoptorch(PopartType);

} // namespace poptorch
