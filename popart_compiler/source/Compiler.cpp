// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <popart/adam.hpp>
#include <popart/adaptive.hpp>
#include <popart/builder.hpp>
#include <popart/error.hpp>
#include <popart/graphtransformer.hpp>
#include <popart/half.hpp>
#include <popart/ir.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/convbase.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensors.hpp>

#include "popart_compiler/Compiler.hpp"
#include "popart_compiler/PopartEnums.hpp"
#include "popart_compiler/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace {

const std::string location_activation = "location_activation";
const std::string location_weight = "location_weight";
const std::string location_optimizer = "location_optimizer";
const std::string location_accumulator = "location_accumulator";

bool ipuModelEnvironmentVariableIsEnabled() {
  if (const char *env_use_model = std::getenv("POPTORCH_IPU_MODEL")) {
    bool model_enabled = std::stoi(env_use_model) != 0;
    logging::info("From POPTORCH_IPU_MODEL environment variable: Ipu model: {}",
                  model_enabled ? "Enabled" : "Disabled");
    return model_enabled;
  }
  return false;
}

bool ipuSmallModelEnvironmentVariableIsEnabled() {
  // POPTORCH_IPU_MODEL takes precedence over the small model.
  if (ipuModelEnvironmentVariableIsEnabled()) {
    return false;
  }
  if (const char *env_use_model = std::getenv("POPTORCH_SMALL_IPU_MODEL")) {
    bool model_enabled = std::stoi(env_use_model) != 0;
    logging::info("From POPTORCH_SMALL_IPU_MODEL environment variable: small "
                  "Ipu model: {}",
                  model_enabled ? "Enabled" : "Disabled");
    return model_enabled;
  }
  return false;
}

std::string getIpuModelVersion() {
  if (const char *env_ipu_model_version =
          std::getenv("POPTORCH_IPU_MODEL_VERSION")) {
    std::string str(env_ipu_model_version);
    return str;
  }
  return "ipu1"; // Default to MK1 if unspecified
}

int getNumTilesPerIpu(const std::string &ipu_model_version) {
  int num_tiles_per_ipu = 0;

  if (ipu_model_version == "ipu1") {
    num_tiles_per_ipu = 1216; // MK1
  }
  if (ipu_model_version == "ipu2") {
    num_tiles_per_ipu = 1472; // MK2
  }

  if (ipuSmallModelEnvironmentVariableIsEnabled()) {
    num_tiles_per_ipu = 4;
  }

  ERROR_ON_MSG(num_tiles_per_ipu == 0,
               "Invalid IPU model version. Valid versions: ipu1, ipu2.");
  return num_tiles_per_ipu;
}

// Wrapper functor used to print to the debug channel the value
// of the options set by poptorch.Options
template <typename Value> class Setter {
public:
  Setter(std::function<void(Value)> fn, std::string name)
      : _fn(std::move(fn)), _name(std::move(name)) {}
  void operator()(Value value);

private:
  std::function<void(Value)> _fn;
  const std::string _name;
};

template <>
void Setter<std::pair<std::string, std::string>>::operator()(
    std::pair<std::string, std::string> value) { // NOLINT
  _fn(value);
  logging::debug("poptorch.Options set {}[{}] to {}", _name, value.first,
                 value.second);
}

template <typename Value> void Setter<Value>::operator()(Value value) {
  _fn(value);
  logging::debug("poptorch.Options set {} to value {}", _name, value);
}

template <typename Value, typename Lambda>
void registerSetter(std::map<std::string, std::function<void(Value)>> &options,
                    const std::string &name, Lambda setter) {
  std::function<void(Value)> fn = setter;
  options[name] = Setter<Value>(fn, name);
}

// Round up the number of IPUs, if required, to the minimum number which need
// to be reservered
std::uint64_t roundUpNumIPUs(std::uint64_t num_ipus) {
  std::uint64_t rounded_num_ipus;

  if (num_ipus < 64) {
    // If fewer than 64, find the next power of 2
    rounded_num_ipus = 1;
    while (rounded_num_ipus < num_ipus) {
      rounded_num_ipus *= 2;
    }
  } else {
    // Otherwise, find the next multiple of 64
    rounded_num_ipus = ((num_ipus - 1) / 64 + 1) * 64;
  }

  return rounded_num_ipus;
}

bool waitIfIpuIsUnavailable() {
  bool wait = false;
  if (const char *env_wait_for_ipu = std::getenv("POPTORCH_WAIT_FOR_IPU")) {
    wait = std::stoi(env_wait_for_ipu) != 0;
    logging::info("From POPTORCH_WAIT_FOR_IPU environment variable: If no IPU "
                  "is available: {}",
                  wait ? "Wait" : "Fail & exit");
  }
  return wait;
}

} // namespace
namespace poptorch {

namespace detail {

enum class ExecutionMode { Pipelined, Sharded, Phased, N };

// To be kept in sync with the Liveness python enum in python/enums.py
enum class Liveness { AlwaysLive, OffChipAfterFwd, OffChipAfterEachPhase, N };

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

class MultiConvBuilder {
public:
  void addConv(const std::vector<popart::TensorId> &inputs,
               const std::vector<int64_t> &dilations,
               const std::vector<int64_t> &kernel_shape,
               const std::vector<int64_t> &pads,
               const std::vector<int64_t> &strides) {
    // Record the inputs and attributes for this single conv
    _inputs.push_back(inputs);
    _dilations.push_back(dilations);
    _kernel_shape.push_back(kernel_shape);
    _pads.push_back(pads);
    _strides.push_back(strides);
  }

  void setAvailableMemoryProportions(const std::vector<float> &v) {
    _options.availableMemoryProportions = v;
  }

  void setPartialsTypes(const std::vector<int64_t> &partials_types) {
    std::vector<std::string> type_strs;

    for (int64_t t : partials_types) {
      if (t == 0) {
        type_strs.emplace_back("float");
      } else if (t == 1) {
        type_strs.emplace_back("half");
      } else {
        ERROR("Invalid MultiConv partials_types");
      }
    }

    _options.partialsTypes = type_strs;
  }

  void setPlanType(int64_t plan_type) {
    if (plan_type == 0) {
      _options.planType = "parallel";
    } else if (plan_type == 1) {
      _options.planType = "serial";
    } else {
      ERROR("Invalid MultiConv plan_type");
    }
  }

  void setPerConvReservedTiles(int n) { _options.perConvReservedTiles = n; }

  void setCycleBackOff(float v) { _options.cycleBackOff = v; }

  std::vector<popart::TensorId> build(popart::Builder *builder) const {
    auto opset = builder->aiGraphcoreOpset1();
    return opset.multiconv(_inputs, _dilations, {}, _pads, {}, _strides,
                           _options.availableMemoryProportions,
                           _options.partialsTypes, _options.planType,
                           _options.perConvReservedTiles,
                           _options.cycleBackOff);
  }

private:
  // Aggregated inputs for all the convs that are fused as a multiconv
  std::vector<std::vector<popart::TensorId>> _inputs;
  std::vector<std::vector<int64_t>> _dilations;
  std::vector<std::vector<int64_t>> _kernel_shape;
  std::vector<std::vector<int64_t>> _pads;
  std::vector<std::vector<int64_t>> _strides;
  popart::MultiConvOptions _options = {{}, {}};
};

struct CompilerImpl {
public:
  friend Compiler;

  CompilerImpl() : op_builder(popart::Builder::create()), loss("") {
    ids.emplace_back(""); // None tensor
    active_builder = op_builder.get();
  }

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

  popart::SessionOptions popart_options;
  struct Options {
    // Number of times the graph will be executed for each execution.
    std::uint64_t steps;
    // Strategy to adopt for returning the graph's output tensors.
    PopartAnchorTypes anchor_mode;
    // 'N' when anchor_mode == PopartAnchorTypes::EveryN
    std::uint64_t anchor_return_period;
    // True if running on the model, False otherwise.
    bool ipu_model;
    // Automatically round up the number of IPUs, if required, to the minimum
    // number required to be reserved
    bool auto_round_num_ipus;
    // Only used for offline compilation (DeviceConnectionType.Never): version
    // of the IPU should the Poplar compiler be targeting.
    std::uint64_t ipu_version;
    // ID of the specific IPU the user wants to use. (If not set we'll just
    // iterate over the IPUs present on the system and try to connect to one
    // that matches our requirements).
    std::uint64_t ipu_id;
    popart::DeviceConnectionType connection_type;
    popart::SyncPattern sync_pattern;
    std::uint64_t random_seed;

    // The frontend will unpack the user option and pass it directly in as
    // [IPU_ID] = Memory proportion for that IPU
    std::unordered_map<std::uint32_t, float> available_memory_proportion;

    // When running in distributed mode: number of processes the training is
    // split// over.
    std::uint64_t num_distributed_processes;
    // In distributed mode: unique ID of this process in [0,
    // num_distributed_processes]// range
    std::uint64_t distributed_process_id;

    popart::Patterns patterns{popart::PatternsLevel::Default};
    ExecutionMode execution_mode;

    // Phased execution options: see the python documentation for more
    // information about how to use them
    //
    // Here is how they translate into Popart options:
    // serial_phases_execution: True -> executionPhaseSettings.stages = 1
    //                          False-> executionPhaseSettings.stages = 2
    //
    // separate_backward_phase:
    //  False:
    //   fwd:       bwd:
    //   phase 0 -> phase 4
    //   phase 1 -> phase 3
    //   phase 2 -> phase 2
    //
    // (End of fwd and start of bwd are part of the same phase)
    //  True:
    //   fwd:       bwd:
    //   phase 0 -> phase 6
    //   phase 1 -> phase 5
    //   phase 2 -> phase 4
    //
    //  This is done by setting options.executionPhaseSettings.phases to N+1
    //
    // tensors_liveness:
    //  Note: tensors have a liveness of [phase, phase+2]
    //  AlwaysLive:
    //   fwd:       bwd:
    //   phase 0 -> phase 6
    //   phase 1 -> phase 5
    //   phase 2 -> phase 4
    //
    //  OffChipAfterFwd:
    //   fwd:       bwd:
    //   phase 0 -> phase 8
    //   phase 1 -> phase 7
    //   phase 2 -> phase 6
    // (Gap between fwd and bwd > 2)
    //  This is done by setting options.executionPhaseSettings.phases to N+2
    //
    //  OffChipAfterEachPhase: (Only for stage=1)
    //   fwd:       bwd:
    //   phase 0 -> phase 20
    //   phase 4 -> phase 16
    //   phase 8 -> phase 12
    // (Gap between each phase > 2)
    //  This is done by setting options.executionPhaseSettings.phases to N+2
    //  and multiplying by 4 the phase_id.
    //
    bool serial_phases_execution;
    bool separate_backward_phase;
    Liveness tensors_liveness;
  };

  // List of options which have been explicitely set by the user.
  std::set<std::string> options_set;

  Options options;

  // We add operations using a state based system so the user would set the
  // active IPU and all subsequent operations will be added to that IPU until
  // stopped.
  std::int64_t active_ipu{0};
  std::uint64_t active_stage{0};
  std::int64_t active_phase{0};
  // Keep track of what the maximum phase number used is.
  std::int64_t max_phase{0};

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

  std::vector<popart::TensorId>
  customOperation(const std::vector<popart::TensorId> &args,
                  const std::string &op, const std::string &domain,
                  std::int64_t version, std::int64_t num_outputs);

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
  std::string checkSystemConfig();
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

  template <typename OptimizerType>
  void updateGroups(OptimizerType *optimizer,
                    const std::vector<Optimizer> &optimizers);

private:
  // Constants which are simply returned (possibly as part of a tuple/list) and
  // do not need to be input into Popart
  std::unordered_map<poptorch::TensorId, HostSideConstant> _host_side_constants;
};

void CompilerImpl::setExecutionStrategyAttributes(
    const std::set<popart::TensorId> &tensors) {
  ERROR_ON_MSG(active_ipu < 0,
               "No active Block, all the ops must belong to a Block");
  switch (options.execution_mode) {
  case ExecutionMode::Pipelined:
  case ExecutionMode::Sharded:
    active_builder->pipelineStage(tensors, active_stage);
    break;
  case ExecutionMode::Phased:
    ERROR_ON(active_phase < 0);
    active_builder->executionPhase(tensors, active_phase);
    break;
  default:
    ERROR("Invalid ExecutionMode active");
  }
  used_ipus.insert(active_ipu);
  active_builder->virtualGraph(tensors, active_ipu);
}

bool WeightsIO::contains(popart::TensorId id) const {
  return _weights.find(id) != _weights.end();
}

popart::MutableVoidData WeightsIO::weight(popart::TensorId id) const {
  return _weights.at(id);
}

void WeightsIO::registerParameter(const popart::TensorId &id,
                                  const popart::TensorInfo &info) {
  ERROR_ON(contains(id));
  _weights[id].info = info;
  _weights_order.push_back(id);
}

void WeightsIO::updateData(const std::vector<void *> &host_buffers) {
  ERROR_ON(host_buffers.size() != _weights_order.size());
  for (std::uint64_t i = 0; i < host_buffers.size(); ++i) {
    _weights[_weights_order[i]].data = host_buffers[i];
  }
}

std::string CompilerImpl::checkSystemConfig() {
  auto dm = popart::DeviceManager::createDeviceManager();
  if (dm.enumerateDevices().empty()) {
    return "\nNo IPU detected in the system: are you sure the gc-driver is "
           "enabled ?";
  }
  if (options_set.count("ipu_id")) {
    return "";
  }
  std::uint64_t num_ipus =
      used_ipus.size() * popart_options.replicatedGraphCount;
  if (dm.enumerateDevices(options.sync_pattern, num_ipus).empty()) {
    std::stringstream ss;
    ss << "\nNo device found on the system with " << num_ipus
       << " IPUs: the configuration needs changing";
    return ss.str();
  }
  return "";
}

void CompilerImpl::updateUseModelConfig() {
  // The configuration set by the application takes precedence over everything
  // else.
  if (options_set.count("ipu_model")) {
    logging::info("From the user configuration: Ipu model: {}",
                  options.ipu_model ? "Enabled" : "Disabled");
  } else if (ipuModelEnvironmentVariableIsEnabled() ||
             ipuSmallModelEnvironmentVariableIsEnabled()) {
    // As a fallback the model can be enabled by the POPTORCH_IPU_MODEL
    // environment variable.
    options.ipu_model = true;
  } else {
    options.ipu_model = false;
  }
}

void CompilerImpl::addMemoryToOutput(poptorch::TensorId id, void *ptr,
                                     std::unique_ptr<popart::IArray> &&memory) {
  if (isHostSideConstant(id)) {
    getHostSideConstant(id).copyDataTo(ptr);
    return;
  }

  memory_manager.push_back(std::move(memory));

  popart::TensorId popart_id = ids[id];
  if (!popart_outgoing.insert({popart_id, *memory_manager.back().get()})
           .second) {
    // Insertion in the map failed because there is already a pointer associated
    // with that id.
    outgoing_duplicates[popart_id].push_back(ptr);
  }
}

void CompilerImpl::addOutputTensor(
    const std::vector<popart::TensorId> &inputs) {
  active_builder->addOutputTensor(inputs.at(0));
}

void CompilerImpl::addInputTensorFromParentGraph(
    const std::vector<popart::TensorId> &inputs) {
  active_builder->addInputTensorFromParentGraph(inputs.at(0));
}

struct SessionOptionsImpl {
  SessionOptionsImpl();

  std::map<std::string, std::function<void(bool)>> bool_options;
  std::map<std::string, std::function<void(std::uint64_t)>> uint64_options;
  std::map<std::string, std::function<void(std::string)>> string_options;
  std::map<std::string, std::function<void(double)>> double_options;

  std::map<std::string,
           std::function<void(std::pair<std::string, std::string>)>>
      container_options;
  std::set<std::string> options_set;

  popart::SessionOptions popart_options;
  CompilerImpl::Options poptorch_options;

  void setMemoryProportion(std::uint32_t ipu, float memory) {
    poptorch_options.available_memory_proportion[ipu] = memory;
  }

  template <typename ValueType>
  void set(const std::string &key, ValueType value,
           std::map<std::string, std::function<void(ValueType)>> &options,
           const std::string &typeStr) {
    auto it = options.find(key);
    ERROR_ON_MSG(it == options.end(),
                 "Unknown " << typeStr << " option " << key);
    it->second(value);
    options_set.insert(key);
  }
};

SessionOptionsImpl::SessionOptionsImpl() {
  // The keys must match the name and type of the attributes of SessionOptions
  // in python/__init__.py

  registerSetter(bool_options, "auto_round_num_ipus", [&](bool value) {
    poptorch_options.auto_round_num_ipus = value;
  });

  registerSetter(bool_options, "use_model",
                 [&](bool value) { poptorch_options.ipu_model = value; });

  registerSetter(bool_options, "serial_phases_execution", [&](bool value) {
    poptorch_options.serial_phases_execution = value;
  });
  registerSetter(bool_options, "separate_backward_phase", [&](bool value) {
    poptorch_options.separate_backward_phase = value;
  });
  registerSetter(uint64_options, "device_iterations",
                 [&](std::uint64_t value) { poptorch_options.steps = value; });
  registerSetter(uint64_options, "num_distributed_processes",
                 [&](std::uint64_t value) {
                   poptorch_options.num_distributed_processes = value;
                 });
  registerSetter(uint64_options, "distributed_process_id",
                 [&](std::uint64_t value) {
                   poptorch_options.distributed_process_id = value;
                 });
  registerSetter(uint64_options, "ipu_version", [&](std::uint64_t value) {
    poptorch_options.ipu_version = value;
  });
  registerSetter(uint64_options, "ipu_id",
                 [&](std::uint64_t value) { poptorch_options.ipu_id = value; });
  registerSetter(
      uint64_options, "gradient_accumulation",
      [&](std::uint64_t value) { popart_options.accumulationFactor = value; });
  registerSetter(uint64_options, "anchor_return_period",
                 [&](std::uint64_t value) {
                   poptorch_options.anchor_return_period = value;
                 });
  registerSetter(uint64_options, "replication_factor",
                 [&](std::uint64_t value) {
                   popart_options.replicatedGraphCount = value;
                 });
  registerSetter(uint64_options, "execution_mode", [&](std::uint64_t value) {
    ERROR_ON_MSG(value >= static_cast<std::uint64_t>(ExecutionMode::N),
                 "Value for ExecutionMode out of range");
    poptorch_options.execution_mode = static_cast<ExecutionMode>(value);
  });
  registerSetter(uint64_options, "tensors_liveness", [&](std::uint64_t value) {
    ERROR_ON_MSG(value >= static_cast<std::uint64_t>(Liveness::N),
                 "Value for Liveness out of range");
    poptorch_options.tensors_liveness = static_cast<Liveness>(value);
  });
  registerSetter(uint64_options, "anchor_mode", [&](std::uint64_t value) {
    ERROR_ON_MSG(value >= static_cast<std::uint64_t>(PopartAnchorTypes::N),
                 "Value for PopartAnchorTypes out of range");
    poptorch_options.anchor_mode = static_cast<PopartAnchorTypes>(value);
  });

  registerSetter(uint64_options, "connection_type", [&](std::uint64_t value) {
    ERROR_ON_MSG(
        value > static_cast<std::uint64_t>(popart::DeviceConnectionType::Never),
        "Value for DeviceConnectionType out of range");
    poptorch_options.connection_type =
        static_cast<popart::DeviceConnectionType>(value);
  });

  registerSetter(
      uint64_options, "accumulateOuterFragmentSettings.schedule",
      [&](std::uint64_t value) {
        ERROR_ON_MSG(
            value > static_cast<std::uint64_t>(
                        popart::AccumulateOuterFragmentSchedule::
                            OverlapMemoryOptimized),
            "Value for popart::AccumulateOuterFragmentSchedule out of range");
        popart_options.accumulateOuterFragmentSettings.schedule =
            static_cast<popart::AccumulateOuterFragmentSchedule>(value);
      });

  registerSetter(container_options,
                 "accumulateOuterFragmentSettings.excludedVirtualGraphs",
                 [&](const std::pair<std::string, std::string> &p) {
                   std::int64_t value = std::stoi(p.first);
                   popart_options.accumulateOuterFragmentSettings
                       .excludedVirtualGraphs.push_back(value);
                 });

  registerSetter(uint64_options, "accumulation_reduction_type",
                 [&](std::uint64_t value) {
                   ERROR_ON_MSG(value > static_cast<std::uint64_t>(
                                            popart::ReductionType::NoReduction),
                                "Value for popart::ReductionType out of range");
                   popart_options.accumulationReductionType =
                       static_cast<popart::ReductionType>(value);
                 });

  registerSetter(uint64_options, "sync_pattern", [&](std::uint64_t value) {
    ERROR_ON_MSG(value > static_cast<std::uint64_t>(
                             popart::SyncPattern::ReplicaAndLadder),
                 "Value for SyncPattern out of range");
    poptorch_options.sync_pattern = static_cast<popart::SyncPattern>(value);
  });

  registerSetter(uint64_options, "random_seed", [&](std::uint64_t value) {
    poptorch_options.random_seed = value;
  });

  registerSetter(string_options, "log_dir", [&](const std::string &value) {
    popart_options.logDir = value;
  });

  string_options["logDir"] = [&](const std::string &log_dir) {
    UNUSED(log_dir);
    logging::warn(
        "Ignoring call to poptorch.Options.Popart.set(\"logDir\",...): use "
        "poptorch.Options.logDir() instead");
  };

  registerSetter(
      container_options, "dotChecks",
      [&](const std::pair<std::string, std::string> &p) {
        std::uint64_t value = std::stoul(p.first);
        ERROR_ON_MSG(value >= static_cast<std::uint64_t>(popart::DotCheck::N),
                     "Value for DotCheck out of range");
        popart_options.dotChecks.insert(static_cast<popart::DotCheck>(value));
      });

  registerSetter(container_options, "hardwareInstrumentations",
                 [&](const std::pair<std::string, std::string> &p) {
                   std::uint64_t value = std::stoul(p.first);
                   ERROR_ON_MSG(value >= static_cast<std::uint64_t>(
                                             popart::Instrumentation::N),
                                "Value for Instrumentation out of range");
                   // clang-format off
                   popart_options.hardwareInstrumentations.insert(
                       static_cast<popart::Instrumentation>(value));
                   // clang-format on
                 });

  registerSetter(container_options, "customCodelets",
                 [&](const std::pair<std::string, std::string> &p) {
                   popart_options.customCodelets.push_back(p.first);
                 });

  registerSetter(container_options, "engineOptions",
                 [&](const std::pair<std::string, std::string> &p) {
                   popart_options.engineOptions.emplace(p);
                 });

  registerSetter(container_options, "reportOptions",
                 [&](const std::pair<std::string, std::string> &p) {
                   popart_options.reportOptions.emplace(p);
                 });

  registerSetter(container_options, "convolutionOptions",
                 [&](const std::pair<std::string, std::string> &p) {
                   popart_options.convolutionOptions.emplace(p);
                 });

  registerSetter(container_options, "gclOptions",
                 [&](const std::pair<std::string, std::string> &p) {
                   popart_options.gclOptions.emplace(p);
                 });

#define ADD_POPART_ENUM_OPTION(name, EnumType)                                 \
  registerSetter(uint64_options, #name, [&](std::uint64_t value) {             \
    ERROR_ON_MSG(value >= static_cast<std::uint64_t>(popart::EnumType::N),     \
                 "Value for " << #EnumType << " out of range");                \
    popart_options.name = static_cast<popart::EnumType>(value);                \
  })

#define ADD_POPART_BOOL_OPTION(name)                                           \
  registerSetter(bool_options, #name,                                          \
                 [&](bool value) { popart_options.name = value; })

#define ADD_POPART_UINT64_OPTION(name)                                         \
  registerSetter(uint64_options, #name,                                        \
                 [&](std::uint64_t value) { popart_options.name = value; })

#define ADD_POPART_DOUBLE_OPTION(name)                                         \
  registerSetter(double_options, #name,                                        \
                 [&](double value) { popart_options.name = value; })

#define ADD_POPART_STRING_OPTION(name)                                         \
  registerSetter(string_options, #name, [&](const std::string &value) {        \
    popart_options.name = value;                                               \
  })

  ADD_POPART_ENUM_OPTION(batchSerializationSettings.transformContext,
                         BatchSerializationTransformContext);
  ADD_POPART_ENUM_OPTION(batchSerializationSettings.method,
                         BatchSerializationMethod);
  ADD_POPART_ENUM_OPTION(batchSerializationSettings.batchSchedule,
                         BatchSerializationBatchSchedule);
  ADD_POPART_ENUM_OPTION(autoRecomputation, RecomputationType);
  ADD_POPART_ENUM_OPTION(mergeVarUpdate, MergeVarUpdateType);
  ADD_POPART_ENUM_OPTION(virtualGraphMode, VirtualGraphMode);
  ADD_POPART_ENUM_OPTION(syntheticDataMode, SyntheticDataMode);
  ADD_POPART_ENUM_OPTION(subgraphCopyingStrategy, SubgraphCopyingStrategy);

  ADD_POPART_STRING_OPTION(logDir);
  ADD_POPART_STRING_OPTION(cachePath);
  ADD_POPART_STRING_OPTION(partialsTypeMatMuls);
  ADD_POPART_STRING_OPTION(customCodeletCompileFlags);
  ADD_POPART_STRING_OPTION(serializedPoprithmsAnnealGraphsDir);
  ADD_POPART_STRING_OPTION(kahnTieBreaker);

  ADD_POPART_UINT64_OPTION(executionPhaseSettings.phases);
  ADD_POPART_UINT64_OPTION(executionPhaseSettings.stages);
  ADD_POPART_UINT64_OPTION(batchSerializationSettings.factor);
  ADD_POPART_UINT64_OPTION(firstDotOp);
  ADD_POPART_UINT64_OPTION(finalDotOp);
  ADD_POPART_UINT64_OPTION(numIOTiles);
  ADD_POPART_UINT64_OPTION(mergeVarUpdateMemThreshold);
  ADD_POPART_UINT64_OPTION(looseThresholdAtPeak);
  ADD_POPART_UINT64_OPTION(accumulationFactor);
  ADD_POPART_UINT64_OPTION(swapLimitScheduler);
  ADD_POPART_UINT64_OPTION(globalReplicationFactor);
  ADD_POPART_UINT64_OPTION(globalReplicaOffset);

  ADD_POPART_BOOL_OPTION(batchSerializationSettings.concatOnVirtualGraphChange);
  ADD_POPART_BOOL_OPTION(
      batchSerializationSettings.concatOnExecutionPhaseChange);
  ADD_POPART_BOOL_OPTION(
      batchSerializationSettings.concatOnPipelineStageChange);
  ADD_POPART_BOOL_OPTION(strictOpVersions);
  ADD_POPART_BOOL_OPTION(opxAliasChecking);
  ADD_POPART_BOOL_OPTION(opxModifyChecking);
  ADD_POPART_BOOL_OPTION(dotOpNames);
  ADD_POPART_BOOL_OPTION(exportPoplarComputationGraph);
  ADD_POPART_BOOL_OPTION(exportPoplarVertexGraph);
  ADD_POPART_BOOL_OPTION(separateCallOpPdfs);
  ADD_POPART_BOOL_OPTION(enableOutlining);
  ADD_POPART_BOOL_OPTION(enableOutliningCopyCostPruning);
  ADD_POPART_BOOL_OPTION(rearrangeAnchorsOnHost);
  ADD_POPART_BOOL_OPTION(enablePrefetchDatastreams);
  ADD_POPART_BOOL_OPTION(enableNonStableSoftmax);
  ADD_POPART_BOOL_OPTION(enableReplicatedGraphs);
  ADD_POPART_BOOL_OPTION(enableGradientAccumulation);
  ADD_POPART_BOOL_OPTION(instrumentWithHardwareCycleCounter);
  ADD_POPART_BOOL_OPTION(enablePipelining);
  ADD_POPART_BOOL_OPTION(disableGradAccumulationTensorStreams);
  ADD_POPART_BOOL_OPTION(compileEngine);
  ADD_POPART_BOOL_OPTION(constantWeights);
  ADD_POPART_BOOL_OPTION(enableEngineCaching);
  ADD_POPART_BOOL_OPTION(enableFloatingPointChecks);
  ADD_POPART_BOOL_OPTION(enableStochasticRounding);
  ADD_POPART_BOOL_OPTION(explicitRecomputation);
  ADD_POPART_BOOL_OPTION(aliasZeroCopy);
  ADD_POPART_BOOL_OPTION(delayVarUpdates);
  ADD_POPART_BOOL_OPTION(enableFullyConnectedPass);
  ADD_POPART_BOOL_OPTION(enableGroupedMatmuls);
  ADD_POPART_BOOL_OPTION(enableSerializedMatmuls);
  ADD_POPART_BOOL_OPTION(enableStableNorm);
  ADD_POPART_BOOL_OPTION(hostAllReduce);
  ADD_POPART_BOOL_OPTION(hostWeightUpdate);
  ADD_POPART_BOOL_OPTION(hostAllReduceRemoteBuffer);
  ADD_POPART_BOOL_OPTION(decomposeGradSum);
  ADD_POPART_BOOL_OPTION(enableDistributedReplicatedGraphs);
  ADD_POPART_BOOL_OPTION(groupHostSync);

  ADD_POPART_DOUBLE_OPTION(outlineSequenceBreakCost);
  ADD_POPART_DOUBLE_OPTION(outlineThreshold);
  ADD_POPART_DOUBLE_OPTION(timeLimitScheduler);

#undef ADD_POPART_STRING_OPTION
#undef ADD_POPART_UINT64_OPTION
#undef ADD_POPART_BOOL_OPTION
#undef ADD_POPART_DOUBLE_OPTION
#undef ADD_POPART_ENUM_OPTION
}

popart::TensorId
CompilerImpl::reshape(const std::vector<popart::TensorId> &inputs,
                      const std::vector<int64_t> &shape) {
  auto ai_onnx = active_builder->aiOnnxOpset10();

  popart::Shape s = {static_cast<int64_t>(shape.size())};
  popart::TensorInfo tensor_info("INT64", s);
  auto new_shape = ai_onnx.constant({shape.data(), tensor_info});
  return ai_onnx.reshape({inputs.at(0), new_shape});
}

std::vector<popart::TensorId>
CompilerImpl::customOperation(const std::vector<popart::TensorId> &args,
                              const std::string &op, const std::string &domain,
                              std::int64_t version, std::int64_t num_outputs) {
  logging::info("Adding operator with {} inputs ",
                static_cast<std::int32_t>(args.size()));

  const std::int32_t num_inputs = static_cast<std::int32_t>(args.size());
  popart::OperatorIdentifier id = {domain, op, 1, num_inputs};

  return active_builder->customOp(id, version, args, num_outputs, {});
}

popart::TensorId CompilerImpl::recomputationCheckpoint(
    const std::vector<popart::TensorId> &inputs) {
  // Popart is simply a for loop over vector inputs and it is better for the
  // PyTorch Graph to avoid Tuple/List packs and unpacks
  ERROR_ON(inputs.size() != 1);
  return active_builder->checkpointOutput(inputs)[0];
}

popart::TensorId
CompilerImpl::tensorConstant(const std::vector<popart::TensorId> &inputs,
                             const PopartConstant &constant) {
  UNUSED(inputs);
  auto ai_onnx = active_builder->aiOnnxOpset10();

  return ai_onnx.constant(*constant.getPopartData());
}

poptorch::TensorId CompilerImpl::hostSideTensorConstant(
    const std::vector<popart::TensorId> &inputs, HostSideConstant constant) {
  UNUSED(inputs);
  _host_side_constants.emplace(std::make_pair(ids.size(), std::move(constant)));

  // Add a dummy into ids
  ids.emplace_back("__poptorch__host_side_constant");

  return ids.size() - 1;
}

std::shared_ptr<popart::DeviceInfo> CompilerImpl::createDevice() {
  updateUseModelConfig();

  std::shared_ptr<popart::DeviceInfo> device;
  std::uint64_t num_ipus =
      used_ipus.size() * popart_options.replicatedGraphCount;
  ERROR_ON_MSG(num_ipus == 0, "Your compiled model is empty (All the "
                              "operations have been optimised out)");
  if (options.ipu_model) {
    if (popart_options.enableEngineCaching) {
      logging::warn("enableExecutableCaching doesn't work with the IPU model");
    }
    std::map<std::string, std::string> model_options;
    model_options["numIPUs"] = std::to_string(num_ipus);
    std::string env_ipu_model_version = getIpuModelVersion();
    model_options["ipuVersion"] = env_ipu_model_version;
    int num_tiles_per_ipu = getNumTilesPerIpu(env_ipu_model_version);
    model_options["tilesPerIPU"] = std::to_string(num_tiles_per_ipu);

    ERROR_ON_MSG(options.connection_type == popart::DeviceConnectionType::Never,
                 "ConnectionType.Never / poptorch.Options.useOfflineIpuTarget "
                 "not supported for the IPU model");
    device = popart::DeviceManager::createDeviceManager().createIpuModelDevice(
        model_options);
    logging::debug("Instantiated device, running on IPU model with {} tiles.",
                   num_tiles_per_ipu);
  } else {
    if (options.connection_type == popart::DeviceConnectionType::Never) {
      // Offline compilation path: create an offline device regardless of what's
      // present on the system.
      ERROR_ON_MSG(options_set.count("ipu_id"),
                   "Offline compilation targeting a specific id not supported");
      std::map<std::string, std::string> device_options;
      device_options["numIPUs"] = std::to_string(num_ipus);
      device_options["ipuVersion"] =
          "ipu" + std::to_string(options.ipu_version);
      device_options["syncPattern"] =
          popart::syncPatternToString(options.sync_pattern);
      device =
          popart::DeviceManager::createDeviceManager().createOfflineIPUDevice(
              device_options);
      ERROR_ON_MSG(!device, "Failed to create offline IPU device");
    } else {
      // Round up number of ipus to a power of 2
      // or a multiple of 64 (number of POD-64s)
      auto rounded_num_ipus = roundUpNumIPUs(num_ipus);

      if (rounded_num_ipus != num_ipus) {
        std::string common_msg(", because PopTorch must reserve a power of 2 or"
                               " a multiple of 64 IPUs.");
        if (options.auto_round_num_ipus) {
          logging::warn("Reserving {} IPUs when the model specifices the use "
                        "of only {}{}. {} will be reserved but not used.",
                        rounded_num_ipus, num_ipus, common_msg,
                        rounded_num_ipus - num_ipus);
          num_ipus = rounded_num_ipus;
        } else {
          ERROR("The model specifies the use of "
                << num_ipus
                << " IPUs, "
                   "however PopTorch must reserve a minimum of "
                << rounded_num_ipus << " in order to allow the model to run"
                << common_msg
                << " Please reconfigure your model to use a "
                   "different number of IPUs or set "
                   "poptorch.Options().autoRoundNumIPUs(True).");
        }
      }

      // Use a lambda to cache the value.
      auto wait_if_unavailable = [this]() {
        // Force disable the wait if the system doesn't contain an IPU that
        // matches the requested config.
        static const bool should_wait =
            waitIfIpuIsUnavailable() && checkSystemConfig().empty();
        return should_wait;
      };

      auto wait_for_a_while = []() {
        constexpr std::int64_t sleep_time = 15;
        logging::trace("No IPU available, sleeping for {} seconds", sleep_time);
        std::this_thread::sleep_for(std::chrono::seconds(sleep_time));
        return true;
      };

      do {
        // Regular IPU hardware target
        if (!options_set.count("ipu_id")) {
          device =
              popart::DeviceManager::createDeviceManager()
                  .acquireAvailableDevice(num_ipus, 0, options.sync_pattern,
                                          options.connection_type);
          ERROR_ON_MSG(!device && !wait_if_unavailable(),
                       "Failed to acquire " << num_ipus << " IPU(s)"
                                            << this->checkSystemConfig());
          if (device) {
            logging::debug("Acquired {} IPU(s): running on device Id {}.",
                           num_ipus, device->getId());
          }
        } else {
          device =
              popart::DeviceManager::createDeviceManager().acquireDeviceById(
                  options.ipu_id, options.sync_pattern,
                  options.connection_type);
          ERROR_ON_MSG(!device && !wait_if_unavailable(),
                       "Failed to acquire device Id " << options.ipu_id
                                                      << checkSystemConfig());
          ERROR_ON_MSG(device && static_cast<std::uint64_t>(
                                     device->getNumIpus()) < num_ipus,
                       "Expected at least replication factor * used IPUs = "
                           << used_ipus.size() << " * "
                           << popart_options.replicatedGraphCount << " = "
                           << num_ipus << " device Ids but the user provided "
                           << device->getNumIpus());
          if (device &&
              static_cast<std::uint64_t>(device->getNumIpus()) != num_ipus) {
            logging::warn(
                "Expected replication factor * used IPUs = {} * {} "
                "= {} device Ids but the device selected has {} IPUs which "
                "means some of them will not be used.",
                used_ipus.size(), popart_options.replicatedGraphCount, num_ipus,
                device->getNumIpus());
          }
          if (device) {
            logging::debug("Acquired IPU device with id {}, running on device.",
                           options.ipu_id);
          }
        }
      } while (!device && wait_for_a_while());
    }
  }
  return device;
}

} // namespace detail

bool ipuHardwareIsAvailable(std::uint64_t num_ipus) {
  return !ipuModelEnvironmentVariableIsEnabled() &&
         !ipuSmallModelEnvironmentVariableIsEnabled() &&
         !popart::DeviceManager::createDeviceManager()
              .enumerateDevices(popart::SyncPattern::Full, num_ipus)
              .empty();
}

namespace {
// Variadic output case. For now we will add all outputs to the graph and
// allocate them on the same IPU but we will only return one. This means only
// one output can be used by user IR (but can still be used by the backed via
// transformations).
template <typename T> struct HandleOutput {
  poptorch::TensorId operator()(T &in, bool loss, detail::CompilerImpl *_impl) {
    ERROR_ON_MSG(loss, "Unreachable internal error: no operation with multiple "
                       "returns is expected to be a loss.");

    std::set<popart::TensorId> ids;

    for (const popart::TensorId &id : in) {
      ids.insert(id);
      _impl->ids.push_back(id);
    }

    _impl->setExecutionStrategyAttributes(ids);

    // Return the first added tensor as the sole return of this IR op.
    return _impl->ids.size() - in.size();
  }
};

// Single tensor output case
template <> struct HandleOutput<popart::TensorId> {
  poptorch::TensorId operator()(const popart::TensorId &in, bool loss,
                                detail::CompilerImpl *_impl) {
    // See if any available memory has been set for this IPU.
    auto itr =
        _impl->options.available_memory_proportion.find(_impl->active_ipu);
    if (itr != _impl->options.available_memory_proportion.end()) {
      logging::info("Setting memory proportion on tensor {} to {}. On IPU {}",
                    in, itr->second, itr->first);
      _impl->active_builder->setAvailableMemoryProportion(in, itr->second);
    }

    _impl->ids.push_back(in);
    _impl->setExecutionStrategyAttributes({in});

    if (loss) {
      _impl->loss = in;
    }

    return _impl->ids.size() - 1;
  }
};

// Host side constant case
template <> struct HandleOutput<poptorch::TensorId> {
  poptorch::TensorId operator()(poptorch::TensorId in, bool loss,
                                detail::CompilerImpl *_impl) {
    UNUSED(loss);
    ERROR_ON(!_impl->isHostSideConstant(in));
    return in;
  }
};

std::string toString(OptimizerType type) {
  switch (type) {
  case OptimizerType::SGD:
    return "SGD";
  case OptimizerType::LAMB:
  case OptimizerType::LAMB_NO_BIAS:
    return "LAMB";
  case OptimizerType::ADAM:
    return "ADAM";
  case OptimizerType::ADAMW:
  case OptimizerType::ADAMW_NO_BIAS:
    return "ADAMW";
  case OptimizerType::RMSPROP_CENTERED:
  case OptimizerType::RMSPROP:
    return "RMSPROP";
  default:
    ERROR("Unreachable: Unsupported optimizer.");
  }
}

// If is_default: return the list of keys accepted by the
// `const std::map<std::string, std::pair<float, bool>> &params` parameter
// of the Popart constructor: it is usually the list of OptimizerValue
// accepted by the explicit constructor.
//
// Else: return the list of keys accepted by insertSpecific (It's usually
// defined in the optimizer's cpp file in a function called getSpecificNames()
// TODO(T33686): these names should be provided by PopART.
std::vector<std::string> getAttributeNames(OptimizerType type,
                                           bool is_default) {
  switch (type) {
  case OptimizerType::SGD: {
    if (is_default) {
      return {"defaultLearningRate",    "defaultWeightDecay",
              "defaultMomentum",        "defaultDampening",
              "defaultVelocityScaling", "lossScaling"};
    }
    return {"learningRate", "weightDecay", "momentum", "dampening",
            "velocityScaling"};
  }
  case OptimizerType::LAMB:
  case OptimizerType::LAMB_NO_BIAS: {
    if (is_default) {
      return {"defaultLearningRate", "defaultWeightDecay", "defaultBeta1",
              "defaultBeta2",        "defaultEps",         "lossScaling",
              "maxWeightNorm"};
    }
    return {"learningRate", "weightDecay", "beta1", "beta2", "eps"};
  }
  case OptimizerType::ADAM:
  case OptimizerType::ADAMW:
  case OptimizerType::ADAMW_NO_BIAS: {
    if (is_default) {
      return {"defaultLearningRate", "defaultWeightDecay", "defaultBeta1",
              "defaultBeta2",        "defaultEps",         "lossScaling"};
    }
    return {"learningRate", "weightDecay", "beta1", "beta2", "eps"};
  }
  case OptimizerType::RMSPROP_CENTERED:
  case OptimizerType::RMSPROP: {
    if (is_default) {
      return {"defaultLearningRate", "defaultWeightDecay", "defaultAlpha",
              "defaultMomentum",     "defaultEps",         "lossScaling"};
    }
    return {"learningRate", "weightDecay", "alpha", "momentum", "eps"};
  }
  default:
    ERROR("Unreachable: Unsupported optimizer.");
  }
}

std::string toString(const std::vector<std::string> &vec) {
  std::stringstream ss;
  ss << "[";
  std::string sep;
  for (const auto &s : vec) {
    ss << sep << s;
    sep = ", ";
  }
  ss << "]";
  return ss.str();
}

int indexOf(const std::vector<std::string> &vec, const std::string &v) {
  auto it = std::find(vec.begin(), vec.end(), v);
  if (it == vec.end()) {
    return -1;
  }
  return it - vec.begin();
}

std::vector<std::string> vectorDiff(const std::vector<std::string> &provided,
                                    const std::vector<std::string> &expected) {
  std::vector<std::string> missing;
  for (const auto &exp : expected) {
    if (indexOf(provided, exp) < 0) {
      missing.push_back(exp);
    }
  }
  return missing;
}

// Convert a Poptorch Optimizer into a map of parameters + types that
// can be understood by the Popart Optimizer / insertSpecific.
struct OptimizerParameters {
public:
  OptimizerParameters(const Optimizer &opt, bool is_default);
  std::string debug() const;
  OptimizerType type;
  bool accum_types_provided;
  popart::DataType accum_type;
  popart::DataType first_order_momentum_accum_type;
  popart::DataType second_order_momentum_accum_type;
  std::map<std::string, std::pair<float, bool>> params;
};

std::string OptimizerParameters::debug() const {
  std::stringstream ss;
  ss << toString(type);
  for (const auto &p : params) {
    ss << ", " << p.first << "=" << p.second.first;
    if (p.second.second) {
      ss << " (const)";
    }
  }
  if (accum_types_provided) {
    ss << ", accumType=" << accum_type;
    ss << ", firstOrderMomentumAccumType=" << first_order_momentum_accum_type;
    ss << ", secondOrderMomentumAccumType=" << second_order_momentum_accum_type;
  }
  return ss.str();
}

OptimizerParameters::OptimizerParameters(const Optimizer &opt, bool is_default)
    : type(opt.type), accum_types_provided(opt.accum_types_provided),
      accum_type(opt.accum_type_is_half ? popart::DataType::FLOAT16
                                        : popart::DataType::FLOAT),
      first_order_momentum_accum_type(
          opt.first_order_momentum_accum_type_is_half
              ? popart::DataType::FLOAT16
              : popart::DataType::FLOAT),
      second_order_momentum_accum_type(
          opt.second_order_momentum_accum_type_is_half
              ? popart::DataType::FLOAT16
              : popart::DataType::FLOAT) {
  // In Popart the attributes which can be specified per group are prefixed with
  // "default" For example learningRate -> defaultLearningRate In order to keep
  // it simple the PopTorch frontend will always use the group name, therefore
  // here we need to remap the PopTorch names to the Popart ones in the default
  // case we then fall back onto the default names for the remaining attributes
  // (e.g lossScaling)
  std::vector<std::string> poptorch_names = getAttributeNames(opt.type, false);
  std::vector<std::string> popart_names =
      getAttributeNames(opt.type, is_default);
  if (is_default) {
    poptorch_names.reserve(popart_names.size());
    for (std::uint64_t i = poptorch_names.size(); i < popart_names.size();
         ++i) {
      poptorch_names.push_back(popart_names[i]);
    }
  }
  std::vector<std::string> provided_names;
  provided_names.reserve(poptorch_names.size());
  for (auto &p : opt.parameters) {
    const std::string name = reinterpret_cast<const char *>(p.name);
    provided_names.push_back(name);
    auto idx = indexOf(poptorch_names, name);
    ERROR_ON_MSG(idx < 0,
                 "Unexpected "
                     << (is_default ? "" : "group ") << "attribute " << name
                     << " for optimizer " << toString(type)
                     << ", allowed values: " << toString(poptorch_names));
    ERROR_ON(
        !params.emplace(popart_names[idx], std::make_pair(p.value, p.is_const))
             .second);
  }
  ERROR_ON_MSG(opt.parameters.size() != poptorch_names.size(),
               "Missing attributes: "
                   << toString(type) << " optimizers require values for "
                   << toString(vectorDiff(provided_names, poptorch_names)));
}

// A whitelist of supported loss operations. Popart needs to know which
// operations are losses so they can be marked by the session.
bool IsLoss(const std::string &operation) {
  return operation == "popart::identityloss";
}

} // namespace

namespace detail {

template <typename OptimizerType>
void CompilerImpl::updateGroups(OptimizerType *optimizer,
                                const std::vector<Optimizer> &optimizers) {
  // For each optimizer group.
  for (std::size_t idx = 1; idx < optimizers.size(); ++idx) {
    // Index 0 is 'defaults'
    const std::size_t group = idx - 1;
    OptimizerParameters group_opt{optimizers[idx], false};
    logging::debug(
        "Updating group {} optimizer with {} for (tensors affected {})", group,
        group_opt.debug(), toString(grad_update_groups[group]));
    // For each tensor in the group.
    for (popart::TensorId &id : grad_update_groups[group]) {
      // Update the optimizer
      optimizer->insertSpecific(id, group_opt.params);
    }
  }
}

std::unique_ptr<popart::Optimizer>
CompilerImpl::getOptimizer(const std::vector<Optimizer> &optimizers) {
  if (optimizers.empty()) {
    return nullptr;
  }

  // The first optimizer contains the default values.
  OptimizerParameters opt{optimizers[0], true};

  // Print to debug the new optimizer.
  logging::debug("Updating graph optimizer with {}", opt.debug());

  switch (opt.type) {
  case OptimizerType::SGD: {
    ERROR_ON(opt.accum_types_provided);
    auto optimizer = std::make_unique<popart::SGD>(opt.params);
    updateGroups(optimizer.get(), optimizers);
    return optimizer;
  }
  case OptimizerType::ADAM:
  case OptimizerType::ADAMW:
  case OptimizerType::ADAMW_NO_BIAS:
  case OptimizerType::LAMB:
  case OptimizerType::LAMB_NO_BIAS: {
    auto adam_mode = popart::AdamMode::AdamNoBias;
    auto decay_mode = popart::WeightDecayMode::Decay;
    if (opt.type == OptimizerType::ADAM) {
      decay_mode = popart::WeightDecayMode::L2Regularization;
    } else if (opt.type == OptimizerType::ADAMW) {
      adam_mode = popart::AdamMode::Adam;
    } else if (opt.type == OptimizerType::LAMB) {
      adam_mode = popart::AdamMode::Lamb;
    } else if (opt.type == OptimizerType::LAMB_NO_BIAS) {
      adam_mode = popart::AdamMode::LambNoBias;
    }

    // NB WeightDecayMode set to default WeightDecayMode::Decay meaning true
    // weight decay rather than L2
    ERROR_ON(!opt.accum_types_provided);
    auto optimizer = std::make_unique<popart::Adam>(
        opt.params, adam_mode, decay_mode, opt.accum_type,
        opt.first_order_momentum_accum_type,
        opt.second_order_momentum_accum_type);
    updateGroups(optimizer.get(), optimizers);
    return optimizer;
  }
  case OptimizerType::RMSPROP:
  case OptimizerType::RMSPROP_CENTERED: {
    ERROR_ON(!opt.accum_types_provided);
    popart::AdaptiveMode mode = opt.type == OptimizerType::RMSPROP
                                    ? popart::AdaptiveMode::RMSProp
                                    : popart::AdaptiveMode::CenteredRMSProp;
    auto optimizer = std::make_unique<popart::Adaptive>(
        opt.params, mode, popart::WeightDecayMode::L2Regularization,
        opt.accum_type, opt.first_order_momentum_accum_type,
        opt.second_order_momentum_accum_type, popart::DataType::FLOAT);
    updateGroups(optimizer.get(), optimizers);
    return optimizer;
  }
  default:
    ERROR("Unreachable: Unsupported optimizer.");
  }
}

popart::TensorId
CompilerImpl::addNotInPlace(const std::vector<popart::TensorId> &in) {
  auto ai_onnx = active_builder->aiOnnxOpset10();
  popart::TensorId output = ai_onnx.add(in);
  active_builder->setInplacePreferences(
      output, {{"AddLhsInplace", -1}, {"AddRhsInplace", -1}});
  return output;
}

popart::TensorId
CompilerImpl::randomNormal(const std::vector<popart::TensorId> &inputs,
                           const std::vector<int64_t> &shape, float mean,
                           float scale, const std::string &dtype) {
  UNUSED(inputs);
  auto ai_onnx = active_builder->aiOnnxOpset10();
  auto pdt = popart::dataTypeFromString(dtype);
  return ai_onnx.randomnormal(shape, popart::getONNXDataTypeAsInt(pdt), mean,
                              scale);
}

popart::TensorId
CompilerImpl::randomUniform(const std::vector<popart::TensorId> &inputs,
                            const std::vector<int64_t> &shape, float high,
                            float low, const std::string &dtype) {
  UNUSED(inputs);
  auto ai_onnx = active_builder->aiOnnxOpset10();
  auto pdt = popart::dataTypeFromString(dtype);
  return ai_onnx.randomuniform(shape, popart::getONNXDataTypeAsInt(pdt), high,
                               low);
}

popart::TensorId CompilerImpl::ones(const std::vector<popart::TensorId> &inputs,
                                    const std::vector<int64_t> &shape,
                                    const std::string &dtype) {
  return zerosOrOnes(inputs, shape, dtype, false);
}

popart::TensorId
CompilerImpl::zeros(const std::vector<popart::TensorId> &inputs,
                    const std::vector<int64_t> &shape,
                    const std::string &dtype) {
  return zerosOrOnes(inputs, shape, dtype, true);
}

popart::TensorId
CompilerImpl::zerosOrOnes(const std::vector<popart::TensorId> &inputs,
                          const std::vector<int64_t> &shape,
                          const std::string &dtype, bool zeros) {
  auto total_size = static_cast<size_t>(std::accumulate(
      shape.begin(), shape.end(), 1, std::multiplies<size_t>()));

  if (dtype == "INT32") {
    std::vector<int32_t> const_buff;
    const_buff.reserve(total_size);
    for (size_t i = 0; i < total_size; i++) {
      const_buff.emplace_back(zeros ? 0 : 1);
    }
    PopartConstant popart_const(PopartType::INT32, &const_buff[0], shape);
    return tensorConstant(inputs, popart_const);
  }
  if (dtype == "FLOAT") {
    std::vector<float> const_buff;
    const_buff.reserve(total_size);
    for (size_t i = 0; i < total_size; i++) {
      const_buff.emplace_back(zeros ? 0 : 1);
    }

    PopartConstant popart_const(PopartType::FLOAT, &const_buff[0], shape);
    return tensorConstant(inputs, popart_const);
  }
  if (dtype == "FLOAT16") {
    std::vector<uint16_t> const_buff;
    const_buff.reserve(total_size);
    for (size_t i = 0; i < total_size; i++) {
      const_buff.emplace_back(popart::floatToHalf(zeros ? 0 : 1));
    }

    PopartConstant popart_const(PopartType::FLOAT16, &const_buff[0], shape);
    return tensorConstant(inputs, popart_const);
  }
  ERROR("Unsupported type " << dtype);
}

const HostSideConstant &
CompilerImpl::getHostSideConstant(poptorch::TensorId id) const {
  return _host_side_constants.at(id);
}

bool CompilerImpl::isHostSideConstant(poptorch::TensorId id) const {
  return _host_side_constants.count(id);
}

void CompilerImpl::addMultiConvPart(const std::vector<popart::TensorId> &inputs,
                                    const std::vector<int64_t> &dilations,
                                    const std::vector<int64_t> &kernel_shape,
                                    const std::vector<int64_t> &pads,
                                    const std::vector<int64_t> &strides) {
  if (multi_conv_builder == nullptr) {
    multi_conv_builder = std::make_unique<MultiConvBuilder>();
  }

  multi_conv_builder->addConv(inputs, dilations, kernel_shape, pads, strides);
}

std::vector<popart::TensorId> CompilerImpl::endMultiConv() {
  ERROR_ON_MSG(multi_conv_builder == nullptr, "Unexpected end_multi_conv.");
  auto outs = multi_conv_builder->build(active_builder);
  multi_conv_builder.reset();
  return outs;
}

} // namespace detail

PopartConstant::PopartConstant(const PopartType &popart_type, const void *data,
                               const std::vector<std::int64_t> &shape) {
  ERROR_ON_MSG(popart_type == PopartType::DOUBLE,
               "Adding a double constant is not supprted. "
               "This should have been demoted to a float");

  popart::TensorInfo info{toPopartTypeStr(popart_type), shape};
  _data = std::make_unique<popart::ConstVoidData>(data, info);
}

PopartConstant::~PopartConstant() = default;

HostSideConstant::HostSideConstant(const PopartType &popart_type, void *data,
                                   size_t data_size,
                                   std::vector<std::int64_t> shape)
    : _popart_type(popart_type), _shape(std::move(shape)) {
  _data.resize(data_size);
  std::memcpy(&_data[0], data, data_size);
}

void HostSideConstant::copyDataTo(void *ptr) const {
  std::memcpy(ptr, &_data[0], _data.size());
}

poptorch::TensorId
Compiler::addInputTensor(const char *type,
                         const std::vector<std::int64_t> &dims) {
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{type, dims};
  _impl->ids.push_back(_impl->active_builder->addInputTensor(info));
  return _impl->ids.size() - 1;
}

#define INT_VEC std::vector<std::int64_t>
#define FLOAT_VEC std::vector<double>
#define FLOAT float
#define INT std::int64_t
#define BOOL bool
#define STRING const char *
#define NONE
#define ARG(Type, Name) , Type Name
#define POPART_CONST_ARG(Name) , const PopartConstant &Name
#define HOST_SIDE_CONST_ARG(Name) , const HostSideConstant &Name
#define BODY_ARG(Name) , Name

// Create a function decl with the given call and arguments.
#define OP_DECL(ns, funcName, function, onnxImpl, Args, BodyArgs)              \
  poptorch::TensorId Compiler::function(                                       \
      const std::vector<poptorch::TensorId> &inputs Args) {                    \
    auto AiOnnxOpset10 = _impl->active_builder->aiOnnxOpset10();               \
    auto AiGraphcoreOpset1 = _impl->active_builder->aiGraphcoreOpset1();       \
    const bool isLoss = IsLoss(#ns "::" #funcName);                            \
    std::vector<popart::TensorId> ins;                                         \
    std::transform(                                                            \
        inputs.begin(), inputs.end(), std::back_inserter(ins),                 \
        [&](poptorch::TensorId index) { return _impl->ids[index]; });          \
    auto output = onnxImpl(ins BodyArgs);                                      \
    return HandleOutput<decltype(output)>{}(output, isLoss, _impl.get());      \
  }

// Create a function decl with the given call and arguments.
#define OP_DECL_NO_RETURN(ns, funcName, function, onnxImpl, Args, BodyArgs)    \
  void Compiler::function(                                                     \
      const std::vector<poptorch::TensorId> &inputs Args) {                    \
    auto AiOnnxOpset10 = _impl->active_builder->aiOnnxOpset10();               \
    auto AiGraphcoreOpset1 = _impl->active_builder->aiGraphcoreOpset1();       \
    std::vector<popart::TensorId> ins;                                         \
    std::transform(                                                            \
        inputs.begin(), inputs.end(), std::back_inserter(ins),                 \
        [&](poptorch::TensorId index) { return _impl->ids[index]; });          \
    onnxImpl(ins BodyArgs);                                                    \
  }

#include "popart_compiler/SupportedOperations.inc.hpp"

#undef OP_DECL
#undef OP_DECL_NO_RETURN
#undef BODY_ARG
#undef POPART_CONST_ARG
#undef HOST_SIDE_CONST_ARG
#undef ARG
#undef NONE
#undef STRING
#undef BOOL
#undef INT
#undef FLOAT
#undef FLOAT_VEC
#undef INT_VEC

poptorch::TensorId
Compiler::addInitializedInputTensor(const char *name, const char *type,
                                    const std::vector<std::int64_t> &dims,
                                    void *data) {
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{type, dims};

  // Create the inital data for the variable.
  popart::ConstVoidData the_data;
  the_data.data = data;
  the_data.info = info;

  _impl->ids.push_back(
      _impl->active_builder->addInitializedInputTensor(the_data, name));

  popart::TensorId id = _impl->ids[_impl->ids.size() - 1];

  _impl->weights.registerParameter(id, info);

  return _impl->ids.size() - 1;
}

void Compiler::addOutputTensor(poptorch::TensorId output) {
  _impl->outputs.push_back(_impl->ids[output]);

  if (isHostSideConstant(output)) {
    return; // Nothing more to do
  }

  const char *as_str = anchorTypeToString(_impl->options.anchor_mode);

  // If we are returning EveryN we need to pass in the return period.
  if (_impl->options.anchor_mode == PopartAnchorTypes::EveryN) {
    _impl->anchors.insert(
        {_impl->ids[output], popart::AnchorReturnType(
                                 as_str, _impl->options.anchor_return_period)});
  } else {
    _impl->anchors.insert(
        {_impl->ids[output], popart::AnchorReturnType(as_str)});
  }
}

void Compiler::setUpInputOp(poptorch::TensorId id, float *ptr,
                            const std::vector<std::int64_t> &dims) {
  assertTensorIs(PopartType::FLOAT, id,
                 static_cast<const char *>(__PRETTY_FUNCTION__));

  // Popart wrapper around the tensor pointer.
  _impl->memory_manager.push_back(
      std::make_unique<popart::NDArrayWrapper<float>>(static_cast<float *>(ptr),
                                                      dims));
  _impl->popart_incoming.insert(
      {_impl->ids[id], *_impl->memory_manager.back().get()});
}

void Compiler::setUpInputOp(poptorch::TensorId id, std::int32_t *ptr,
                            const std::vector<std::int64_t> &dims) {
  assertTensorIs(PopartType::INT32, id,
                 static_cast<const char *>(__PRETTY_FUNCTION__));

  // Popart wrapper around the tensor pointer.
  _impl->memory_manager.push_back(
      std::make_unique<popart::NDArrayWrapper<std::int32_t>>(ptr, dims));
  _impl->popart_incoming.insert(
      {_impl->ids[id], *_impl->memory_manager.back().get()});
}

void Compiler::setUpInputOp(poptorch::TensorId id, bool *ptr,
                            const std::vector<std::int64_t> &dims) {
  assertTensorIs(PopartType::BOOL, id,
                 static_cast<const char *>(__PRETTY_FUNCTION__));

  // Popart wrapper around the tensor pointer.
  _impl->memory_manager.push_back(
      std::make_unique<popart::NDArrayWrapper<bool>>(ptr, dims));
  _impl->popart_incoming.insert(
      {_impl->ids[id], *_impl->memory_manager.back().get()});
}

void Compiler::setUpInputOp(poptorch::TensorId id, std::int16_t *ptr,
                            const std::vector<std::int64_t> &dims,
                            bool float16) {
  if (float16) {
    assertTensorIs(PopartType::FLOAT16, id,
                   static_cast<const char *>(__PRETTY_FUNCTION__));
  } else {
    assertTensorIs(PopartType::INT16, id,
                   static_cast<const char *>(__PRETTY_FUNCTION__));
  }

  // Popart wrapper around the tensor pointer.
  _impl->memory_manager.push_back(
      std::make_unique<popart::NDArrayWrapper<std::int16_t>>(
          ptr, popart::TensorInfo(float16 ? popart::DataType::FLOAT16
                                          : popart::DataType::INT16,
                                  dims)));
  _impl->popart_incoming.insert(
      {_impl->ids[id], *_impl->memory_manager.back().get()});
}

void Compiler::setUpOutputOp(poptorch::TensorId id, float *ptr,
                             const std::vector<std::int64_t> &dims) {
  // Popart wrapper around the tensor pointer.
  auto memory = std::make_unique<popart::NDArrayWrapper<float>>(
      static_cast<float *>(ptr), dims);

  _impl->addMemoryToOutput(id, ptr, std::move(memory));
}

void Compiler::setUpOutputOp(poptorch::TensorId id, std::int32_t *ptr,
                             const std::vector<std::int64_t> &dims) {
  // Popart wrapper around the tensor pointer.
  auto memory = std::make_unique<popart::NDArrayWrapper<std::int32_t>>(
      static_cast<std::int32_t *>(ptr), dims);

  _impl->addMemoryToOutput(id, ptr, std::move(memory));
}

void Compiler::setUpOutputOp(poptorch::TensorId id, bool *ptr,
                             const std::vector<std::int64_t> &dims) {
  // Popart wrapper around the tensor pointer.
  auto memory = std::make_unique<popart::NDArrayWrapper<bool>>(
      static_cast<bool *>(ptr), dims);

  _impl->addMemoryToOutput(id, ptr, std::move(memory));
}

void Compiler::setUpOutputOp(poptorch::TensorId id, std::int16_t *ptr,
                             const std::vector<std::int64_t> &dims) {
  // Popart wrapper around the tensor pointer.
  auto memory = std::make_unique<popart::NDArrayWrapper<std::int16_t>>(
      static_cast<std::int16_t *>(ptr), dims);

  _impl->addMemoryToOutput(id, ptr, std::move(memory));
}

void Compiler::initSession(const std::vector<Optimizer> &optimizers) {
  auto device = _impl->createDevice();
  popart::SessionOptions &options = _impl->popart_options;

  // If Pipelining wasn't set: enable it if more than 1 IPU is used.
  switch (_impl->options.execution_mode) {
  case detail::ExecutionMode::Pipelined: {
    _impl->setOptionIfNotSet(options.enablePipelining,
                             _impl->used_ipus.size() > 1, "enablePipelining");
    _impl->setOptionIfNotSet(
        options.virtualGraphMode, popart::VirtualGraphMode::Manual,
        "virtualGraphMode", popart::toString(options.virtualGraphMode));
    // If we are pipelining we want to turn on recompute by default.
    if (_impl->used_ipus.size() > 1) {
      _impl->setOptionIfNotSet(
          options.autoRecomputation, popart::RecomputationType::Pipeline,
          "autoRecomputation",
          popart::toString(popart::RecomputationType::Pipeline));
    }

    break;
  }
  case detail::ExecutionMode::Sharded: {
    _impl->setOptionIfNotSet(options.enablePipelining, false,
                             "enablePipelining");
    _impl->setOptionIfNotSet(
        options.virtualGraphMode, popart::VirtualGraphMode::Manual,
        "virtualGraphMode", popart::toString(options.virtualGraphMode));
    break;
  }
  case detail::ExecutionMode::Phased: {
    _impl->setOptionIfNotSet(options.enablePipelining, false,
                             "enablePipelining");
    _impl->setOptionIfNotSet(
        options.virtualGraphMode, popart::VirtualGraphMode::ExecutionPhases,
        "virtualGraphMode", popart::toString(options.virtualGraphMode));
    std::uint64_t num_phases = _impl->max_phase + 1;
    std::uint64_t num_stages;
    if (_impl->options.tensors_liveness != detail::Liveness::AlwaysLive) {
      // We want to send the tensors off chip: we need to have a gap of N+2
      // before the backward pass.
      num_phases += 2;
    } else if (_impl->options.separate_backward_phase) {
      // Make sure the backward pass will start with a new phase.
      num_phases += 1;
    }
    if (_impl->options.serial_phases_execution) {
      num_stages = 1;
    } else {
      num_stages = 2;
    }
    _impl->setOptionIfNotSet(options.executionPhaseSettings.phases, num_phases,
                             "executionPhaseSettings.phases");
    _impl->setOptionIfNotSet(options.executionPhaseSettings.stages, num_stages,
                             "executionPhaseSettings.stages");
    _impl->setOptionIfNotSet(
        options.activationTensorLocationSettings.location.storage,
        popart::TensorStorage::OffChip, location_activation,
        "useOnChipStorage(False)");
    _impl->setOptionIfNotSet(
        options.weightTensorLocationSettings.location.storage,
        popart::TensorStorage::OffChip, location_weight,
        "useOnChipStorage(False)");
    _impl->setOptionIfNotSet(
        options.optimizerStateTensorLocationSettings.location.storage,
        popart::TensorStorage::OffChip, location_optimizer,
        "useOnChipStorage(False)");
    _impl->setOptionIfNotSet(
        options.accumulatorTensorLocationSettings.location.storage,
        popart::TensorStorage::OffChip, location_accumulator,
        "useOnChipStorage(False)");
    break;
  }
  default:
    ERROR("ExecutionMode not supported");
  }

  _impl->setOptionIfNotSet(options.enableDistributedReplicatedGraphs,
                           _impl->options.num_distributed_processes > 1,
                           "enableDistributedReplicatedGraphs");

  _impl->setOptionIfNotSet(options.globalReplicationFactor,
                           _impl->options.num_distributed_processes *
                               options.replicatedGraphCount,
                           "globalReplicationFactor");
  _impl->setOptionIfNotSet(options.globalReplicaOffset,
                           _impl->options.distributed_process_id *
                               options.replicatedGraphCount,
                           "globalReplicaOffset");

  _impl->setOptionIfNotSet(options.enableReplicatedGraphs,
                           options.replicatedGraphCount > 1,
                           "enableReplicatedGraphs");

  // Disable constant_weights by default: causes problems with Popart
  _impl->setOptionIfNotSet(options.constantWeights, false, "constantWeights");

  if (_impl->options.execution_mode == detail::ExecutionMode::Pipelined) {
    ERROR_ON_MSG(_impl->used_ipus.size() > popartBatchDim(),
                 "poptorch.Options.deviceIterations("
                     << _impl->options.steps
                     << ") * poptorch.Options.gradientAccumulation("
                     << _impl->popart_options.accumulationFactor
                     << ") * poptorch.Options.replicationFactor("
                     << _impl->popart_options.replicatedGraphCount
                     << ") = " << popartBatchDim()
                     << " must be greater or equal than the number of IPUs used"
                        " by the model: "
                     << _impl->used_ipus.size());
  }

  _impl->setOptionIfNotSet(options.enableGradientAccumulation,
                           options.accumulationFactor > 1,
                           "enableGradientAccumulation");

  // Create the anchors, these are used to copy to the host.
  auto data_flow = popart::DataFlow(_impl->options.steps, _impl->anchors);

  // Create the popart session object to actually run the graph.
  if (!_impl->is_training) {
    // Create an inference session.
    logging::LogContext ctx{
        "Compiler::initSession popart::InferenceSession::createFromOnnxModel"};
    _impl->session = popart::InferenceSession::createFromOnnxModel(
        _impl->active_builder->getModelProto(), data_flow, device, {}, options,
        popart::PatternsLevel::Default);
  } else {
    // Create the optimizer from user provided parameters.
    std::unique_ptr<popart::Optimizer> optimizer =
        _impl->getOptimizer(optimizers);

    // Transform nodes which have training/inference variants. I.E BatchNorm.
    popart::GraphTransformer transformer{
        _impl->active_builder->getModelProto()};
    transformer.prepareNodesForTraining();

    // Create the training session.
    logging::LogContext ctx{
        "Compiler::initSession popart::TrainingSession::createFromOnnxModel"};
    _impl->session = popart::TrainingSession::createFromOnnxModel(
        transformer.getModelProto(), data_flow, _impl->loss, *optimizer, device,
        {}, options, _impl->options.patterns);
  }

  // Poplar compilation.
  try {
    logging::LogContext ctx{
        "Compiler::initSession popart::Session::prepareDevice: Poplar "
        "compilation"};
    logging::trace("Begining Poplar compilation.");
    _impl->session->prepareDevice();
    logging::trace("Finished Poplar compilation.");

    // serializeIr must be called after prepareDevice in some cases (e.g.
    // when loading from execution cache)
    logging::trace(
        "Popart serialised IR:\n{}",
        _impl->session->serializeIr(popart::IrSerializationFormat::JSON));
  } catch (popart::memory_allocation_err &e) {
    std::ofstream stream;
    stream.open("OOMReport.json");
    stream << e.getGraphReport(true);
    stream.close();

    std::rethrow_exception(std::current_exception());
  }

  // Set the random seed (if one was provided) following compilation
  if (_impl->options_set.count("random_seed")) {
    logging::trace("Setting random seed to: {}", _impl->options.random_seed);
    _impl->session->setRandomSeed(_impl->options.random_seed);
  }
}

std::unique_ptr<char[]> Compiler::getExecutionInfo() const {
  std::stringstream info;
  switch (_impl->options.execution_mode) {
  case detail::ExecutionMode::Pipelined: {
    info << " mode(Pipelined), ipu(" << _impl->active_ipu << "), stage("
         << _impl->active_stage << ")";
    break;
  }
  case detail::ExecutionMode::Sharded: {
    info << " mode(Sharded), ipu(" << _impl->active_ipu << "), stage("
         << _impl->active_stage << ")";
    break;
  }
  case detail::ExecutionMode::Phased: {
    info << " mode(Phased), ipu(" << _impl->active_ipu << "), phase("
         << _impl->active_phase << ")";
    break;
  }
  default:
    ERROR("Invalid ExecutionMode active");
  }
  const std::string as_string = info.str();

  // Copy into a memory managed array to get around ABI.
  return stringToUniquePtr(as_string);
}

std::unique_ptr<char[]> Compiler::getPopartIR() const {
  const std::string as_string =
      _impl->session->serializeIr(popart::IrSerializationFormat::JSON);

  // Copy into a memory managed array to get around ABI.
  return stringToUniquePtr(as_string);
}

// Write the weights into IPU memory from the pytorch tensor buffers in the
// model.
void Compiler::copyWeightsToDevice(const std::vector<void *> &host_buffers) {
  logging::info("Writing weights from host to IPU memory.");
  _impl->weights.updateData(host_buffers);
  _impl->session->writeWeights(_impl->weights);
  _impl->session->weightsFromHost();
}

// Read the weights from IPU memory into the pytorch tensor buffers.
void Compiler::copyWeightsToHost(const std::vector<void *> &host_buffers) {
  logging::info("Writing weights from IPU to host.");
  _impl->session->weightsToHost();
  _impl->weights.updateData(host_buffers);
  _impl->session->readWeights(_impl->weights);
}

void Compiler::run(const std::vector<Optimizer> &optimizers) {
  if (!optimizers.empty() && _impl->is_training) {
    std::unique_ptr<popart::Optimizer> optimizer =
        _impl->getOptimizer(optimizers);

    // Update the popart graph/poplar executable with the new optimizer.
    popart::TrainingSession &session =
        dynamic_cast<popart::TrainingSession &>(*_impl->session);
    session.updateOptimizerFromHost(optimizer.get());
  }

  // Execute the model on IPU.
  popart::StepIO stepio(_impl->popart_incoming, _impl->popart_outgoing);
  _impl->session->run(stepio);

  // In case several outputs point at the same tensor: duplicate the data
  for (const auto &out : _impl->outgoing_duplicates) {
    auto &src = _impl->popart_outgoing.at(out.first);
    for (auto ptr : out.second) {
      std::memcpy(ptr, src.data(),
                  src.nelms() *
                      popart::getDataTypeInfoMap().at(src.dataType()).nbytes());
    }
  }
  // The buffers handle the communication between pytorch and popart, we set
  // them up each run.
  _impl->popart_incoming.clear();
  _impl->popart_outgoing.clear();
  _impl->outgoing_duplicates.clear();
  _impl->memory_manager.clear();
}

poptorch::PopartType Compiler::getPopartType(poptorch::TensorId id) const {
  if (isHostSideConstant(id)) {
    return _impl->getHostSideConstant(id).popartType();
  }

  popart::TensorInfo info = _impl->session->getInfo(_impl->ids[id]);

#define DEFINE_CASE(value)                                                     \
  case popart::DataType::value: {                                              \
    return PopartType::value;                                                  \
  }

  switch (info.dataType()) { FOR_ALL_POPART_TYPES(DEFINE_CASE) }
#undef DEFINE_CASE

  ERROR("Unsupported popart type in return: " << info.data_type());
}

const char *Compiler::tensorName(poptorch::TensorId id) const {
  return _impl->ids.at(id).c_str();
}

bool Compiler::tensorIdIsValid(poptorch::TensorId id) const {
  return id < _impl->ids.size();
}

std::vector<std::int64_t> Compiler::getSize(poptorch::TensorId id) const {
  if (isHostSideConstant(id)) {
    return _impl->getHostSideConstant(id).shape();
  }

  if (_impl->session) {
    return _impl->session->getInfo(_impl->ids[id]).shape();
  }

  auto popart_id = _impl->ids.at(id);
  try {
    return _impl->active_builder->getTensorShape(popart_id);
  } catch (const popart::error &e) {
    return {};
  }
}

std::unique_ptr<char[]>
Compiler::getTensorDTypeString(poptorch::TensorId id) const {
  std::string type_str;

  if (_impl->session) {
    type_str = _impl->session->getInfo(_impl->ids[id]).data_type();
  } else {
    auto popart_id = _impl->ids.at(id);
    try {
      type_str = _impl->active_builder->getTensorDtypeString(popart_id);
    } catch (const popart::error &e) {
      type_str = "unknown";
    }
  }

  return stringToUniquePtr(type_str);
}

void Compiler::clearActiveIpu() { _impl->active_ipu = -1; }

void Compiler::setActiveIpu(std::uint64_t stage_id, std::int64_t phase_id,
                            std::int64_t ipu_id) {
  switch (_impl->options.execution_mode) {
  case detail::ExecutionMode::Phased:
    ERROR_ON_MSG(phase_id < 0, "Invalid phase for ExecutionMode::Phased");
    if (_impl->options.tensors_liveness ==
        detail::Liveness::OffChipAfterEachPhase) {
      ERROR_ON_MSG(!_impl->options.serial_phases_execution,
                   "This is only supported for serial phase execution");
      _impl->active_phase = phase_id * 4;
    } else {
      _impl->active_phase = phase_id;
      _impl->max_phase = std::max(phase_id, _impl->max_phase);
    }
    if (!_impl->options.serial_phases_execution) {
      ERROR_ON_MSG(_impl->active_phase % 2 != ipu_id % 2,
                   "When phases are executed in parallel: even phases must run "
                   "on even IPUs and odd phases on odd IPUs");
    }
    break;
  case detail::ExecutionMode::Pipelined:
  case detail::ExecutionMode::Sharded:
    _impl->active_stage = stage_id;
    break;
  default:
    ERROR("Unsupported ExecutionMode");
  }
  _impl->active_ipu = ipu_id;
}

bool Compiler::isHostSideConstant(poptorch::TensorId id) const {
  return _impl->isHostSideConstant(id);
}

std::uint64_t Compiler::batchPerStep() const { return _impl->options.steps; }

std::uint64_t Compiler::popartBatchDim() const {
  return _impl->popart_options.replicatedGraphCount * _impl->options.steps *
         _impl->popart_options.accumulationFactor;
}

std::uint64_t Compiler::popartBatchDimForAnchor(poptorch::TensorId id) const {
  if (isHostSideConstant(id)) {
    return 1; // Cannot be batched as it is a constant
  }

  // Get the PopART tensor from our wrapper.
  popart::TensorId popart_id = _impl->ids[id];

  // Check what the anchor is supposed to return.
  auto iterator = _impl->anchors.find(popart_id);
  ERROR_ON_MSG(iterator == _impl->anchors.end(),
               "Internal Error: Output op doesn't have an anchor.");

  const popart::AnchorReturnType &return_type = iterator->second;

  // If we are returning ALL then we are returning a full batch.
  if (return_type.id() == popart::AnchorReturnTypeId::All) {
    return popartBatchDim();
  }

  // If we are copying EveryN then we will be returning N.
  if (return_type.id() == popart::AnchorReturnTypeId::EveryN) {
    return popartBatchDim() / return_type.rp();
  }

  // Return an element for each replica.
  return _impl->popart_options.replicatedGraphCount;
}

void Compiler::setAvailableMemoryProportion(
    const std::vector<poptorch::TensorId> &inputs,
    float availableMemoryProportion) {
  for (const poptorch::TensorId &id : inputs) {
    _impl->active_builder->setAvailableMemoryProportion(
        _impl->ids[id], availableMemoryProportion);
  }
}

void Compiler::setMatMulSerialization(poptorch::TensorId matmul,
                                      const char *mode, std::uint64_t factor,
                                      std::uint64_t keep_precision) {
  _impl->active_builder->setSerializeMatMul({_impl->ids[matmul]}, mode, factor,
                                            keep_precision);
}

void Compiler::optimizerGroup(const std::vector<poptorch::TensorId> &inputs,
                              int64_t group) {
  _impl->optimizerGroup(inputs, group);
}

Compiler::Compiler(Compiler &&compiler) { _impl = std::move(compiler._impl); }

Compiler::Compiler(bool is_training, const SessionOptions &options) {
  _impl = std::make_unique<detail::CompilerImpl>();
  _impl->is_training = is_training;
  _impl->popart_options = options._impl->popart_options;
  _impl->options = options._impl->poptorch_options;
  _impl->options_set = options._impl->options_set;
}

Compiler::~Compiler() = default;

void Compiler::addOutputType(OutputType type) {
  _impl->output_types.emplace_back(type);
}

const std::vector<OutputType> &Compiler::outputTypes() const {
  return _impl->output_types;
}

void Compiler::startSubgraph() {
  popart::Builder *subgraph = &_impl->active_builder->createSubgraphBuilder();
  _impl->active_builder = subgraph;

  _impl->active_builder->addInputTensor(
      popart::TensorInfo{"INT64", popart::Shape{}});
  popart::TensorId keep_going = _impl->active_builder->addInputTensor(
      popart::TensorInfo{"BOOL", popart::Shape{}});

  _impl->active_builder->addOutputTensor({keep_going});
}

void Compiler::startIfBlock() {
  popart::Builder *subgraph = &_impl->active_builder->createSubgraphBuilder();
  _impl->active_builder = subgraph;
  _impl->if_true_stack.push(_impl->active_builder);
}

void Compiler::startElseBlock() {
  // Else must by definition be added after a if block.
  _impl->active_builder = _impl->active_builder->getParent();
  popart::Builder *subgraph = &_impl->active_builder->createSubgraphBuilder();
  _impl->active_builder = subgraph;
  _impl->if_false_stack.push(_impl->active_builder);
}

poptorch::TensorId
Compiler::endForLoop(std::int32_t trip_count, std::int64_t num_outputs,
                     const std::vector<poptorch::TensorId> &inputs) {
  popart::Builder *body = _impl->active_builder;

  // Switch back to main graph.
  _impl->active_builder = _impl->active_builder->getParent();
  auto ai_onnx = _impl->active_builder->aiOnnxOpset10();

  PopartConstant popart_const(PopartType::INT32, &trip_count, {});
  popart::TensorId trip_count_as_tensor =
      _impl->tensorConstant({}, popart_const);

  popart::ConstVoidData the_data;

  const bool true_const = true;
  the_data.data = &true_const;
  the_data.info = {"BOOL", popart::Shape{}};

  popart::TensorId condition = ai_onnx.constant(the_data);

  std::vector<popart::TensorId> transformed_ins = {trip_count_as_tensor,
                                                   condition};

  for (poptorch::TensorId id : inputs) {
    transformed_ins.push_back(_impl->ids[id]);
  }

  std::vector<popart::TensorId> output =
      ai_onnx.loop(transformed_ins, num_outputs, *body);

  for (const popart::TensorId &id : output) {
    logging::warn("Output {}", id);
  }

  return HandleOutput<std::vector<popart::TensorId>>{}(output, false,
                                                       _impl.get());
}

poptorch::TensorId Compiler::endIf(const poptorch::TensorId &condition,
                                   std::size_t num_outputs) {
  // Pop back to the parent.
  _impl->active_builder = _impl->active_builder->getParent();

  // Pop the true branch off the stack.
  popart::Builder *true_branch = _impl->if_true_stack.top();
  _impl->if_true_stack.pop();

  // Pop the false branch off the stack.
  popart::Builder *false_branch = _impl->if_false_stack.top();
  _impl->if_false_stack.pop();

  popart::TensorId cond_as_popart =
      _impl->convertPoptorchToPopartTensor(condition);

  auto ai_onnx = _impl->active_builder->aiOnnxOpset10();
  std::vector<popart::TensorId> output = ai_onnx.logical_if(
      {cond_as_popart}, num_outputs, *false_branch, *true_branch);
  return HandleOutput<std::vector<popart::TensorId>>{}(output, false,
                                                       _impl.get());
}

poptorch::TensorId Compiler::addUntypedInputTensor() {
  popart::TensorId out = _impl->active_builder->addUntypedInputTensor();
  _impl->ids.push_back(out);
  return _impl->ids.size() - 1;
}

void Compiler::assertTensorIs(PopartType dataType, poptorch::TensorId id,
                              const char *caller) const {
  PopartType actual_type;
  try {
    actual_type = getPopartType(id);
  } catch (const popart::error &) {
    // Rare case of input tensor never used, so not in IR
    return;
  }

  ERROR_ON_MSG(actual_type != dataType,
               "Incorrect type for tensor, " << id << " used in " << caller);
}

void Compiler::addMultiConvPart(const std::vector<poptorch::TensorId> &inputs,
                                const std::vector<int64_t> &dilations,
                                const std::vector<int64_t> &kernel_shape,
                                const std::vector<int64_t> &pads,
                                const std::vector<int64_t> &strides) {
  std::vector<popart::TensorId> args;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(args),
                 [&](poptorch::TensorId index) { return _impl->ids[index]; });
  _impl->addMultiConvPart(args, dilations, kernel_shape, pads, strides);
}

void Compiler::setMultiConvAvailableMemoryProportions(
    const std::vector<double> &v) {
  ERROR_ON_MSG(
      _impl->multi_conv_builder == nullptr,
      "Unexpected poptorch.MultiConv option: available_memory_proportions");
  _impl->multi_conv_builder->setAvailableMemoryProportions(
      popart::vXtoY<double, float>(v));
}

void Compiler::setMultiConvPartialsTypes(
    const std::vector<int64_t> &partials_types) {
  ERROR_ON_MSG(_impl->multi_conv_builder == nullptr,
               "Unexpected poptorch.MultiConv option: partials_types");
  _impl->multi_conv_builder->setPartialsTypes(partials_types);
}

void Compiler::setMultiConvPlanType(int64_t plan_type) {
  ERROR_ON_MSG(_impl->multi_conv_builder == nullptr,
               "Unexpected poptorch.MultiConv option: plan_type");
  _impl->multi_conv_builder->setPlanType(plan_type);
}

void Compiler::setMultiConvPerConvReservedTiles(int64_t v) {
  ERROR_ON_MSG(_impl->multi_conv_builder == nullptr,
               "Unexpected poptorch.MultiConv option: per_conv_reserved_tiles");
  _impl->multi_conv_builder->setPerConvReservedTiles(static_cast<int>(v));
}

void Compiler::setMultiConvCycleBackOff(double c) {
  ERROR_ON_MSG(_impl->multi_conv_builder == nullptr,
               "Unexpected poptorch.MultiConv option: cycle_back_off");
  _impl->multi_conv_builder->setCycleBackOff(static_cast<float>(c));
}

std::vector<poptorch::TensorId> Compiler::endMultiConv() {
  auto outputs = _impl->endMultiConv();
  poptorch::TensorId first =
      HandleOutput<decltype(outputs)>{}(outputs, false, _impl.get());
  std::vector<poptorch::TensorId> out_ids(outputs.size());
  std::iota(out_ids.begin(), out_ids.end(), first);
  return out_ids;
}

SessionOptions::SessionOptions()
    : _impl(std::make_unique<detail::SessionOptionsImpl>()) {}

SessionOptions::SessionOptions(SessionOptions &&src)
    : _impl(std::move(src._impl)) {}

void SessionOptions::addStringOption(const char *option, const char *value) {
  _impl->set<std::string>(option, value, _impl->string_options, "string");
}

void SessionOptions::addUint64Option(const char *option, std::uint64_t value) {
  _impl->set(option, value, _impl->uint64_options, "uint64");
}

void SessionOptions::addBoolOption(const char *option, bool value) {
  _impl->set(option, value, _impl->bool_options, "bool");
}

void SessionOptions::addDoubleOption(const char *option, double value) {
  _impl->set(option, value, _impl->double_options, "floating point");
}

void SessionOptions::insertStringOption(const char *option, const char *value) {
  _impl->set(option, std::pair<std::string, std::string>(value, ""),
             _impl->container_options, "set / vector");
}

void SessionOptions::insertStringPairOption(const char *option, const char *key,
                                            const char *value) {
  _impl->set(option, std::pair<std::string, std::string>(key, value),
             _impl->container_options, "map");
}

void SessionOptions::setMemoryProportion(std::uint32_t ipu, float memory) {
  _impl->setMemoryProportion(ipu, memory);
}

void SessionOptions::setPatternsLevel(std::uint64_t level) {
  _impl->options_set.insert("patterns");
  ERROR_ON(level > static_cast<std::uint64_t>(popart::PatternsLevel::All));
  _impl->poptorch_options.patterns =
      popart::Patterns(static_cast<popart::PatternsLevel>(level));
}

void SessionOptions::addPattern(const char *pattern, bool enabled) {
  _impl->poptorch_options.patterns.enablePattern(pattern, enabled);
}

void SessionOptions::setTensorLocation(const char *tensor, const char *option,
                                       std::uint64_t value) {
  logging::debug("Setting {} to {} for location {}", option, value, tensor);
  std::string location_tensor{tensor};
  std::string opt{option};
  popart::TensorLocationSettings *settings;
  _impl->options_set.insert(location_tensor);
  if (location_tensor == location_activation) {
    settings = &_impl->popart_options.activationTensorLocationSettings;
  } else if (location_tensor == location_weight) {
    settings = &_impl->popart_options.weightTensorLocationSettings;
  } else if (location_tensor == location_optimizer) {
    settings = &_impl->popart_options.optimizerStateTensorLocationSettings;
  } else if (location_tensor == location_accumulator) {
    settings = &_impl->popart_options.accumulatorTensorLocationSettings;
  } else {
    ERROR("Unknown tensor location " << location_tensor);
  }

  if (opt == "minElementsForOffChip") {
    settings->minElementsForOffChip = value;
  } else if (opt == "minElementsForReplicatedTensorSharding") {
    settings->minElementsForReplicatedTensorSharding = value;
  } else if (opt == "onChip") {
    settings->location.storage = value > 0 ? popart::TensorStorage::OnChip
                                           : popart::TensorStorage::OffChip;
  } else if (opt == "useReplicatedTensorSharding") {
    settings->location.replicatedTensorSharding =
        value > 0 ? popart::ReplicatedTensorSharding::On
                  : popart::ReplicatedTensorSharding::Off;
  } else if (opt == "useIOTilesToLoad") {
    settings->location.loadTileSet =
        value > 0 ? popart::TileSet::IO : popart::TileSet::Compute;
  } else if (opt == "useIOTilesToStore") {
    settings->location.storageTileSet =
        value > 0 ? popart::TileSet::IO : popart::TileSet::Compute;
  } else {
    ERROR("Unknown option '" << opt << "' for tensor location "
                             << location_tensor);
  }
}

SessionOptions::~SessionOptions() = default;
} // namespace poptorch
