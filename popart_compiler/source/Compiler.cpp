// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <popart/builder.hpp>
#include <popart/graphtransformer.hpp>
#include <popart/ir.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/session.hpp>
#include <popart/tensors.hpp>
#include <poptorch_logging/Error.hpp>
#include <poptorch_logging/Logging.hpp>

#include <popart_compiler/Compiler.hpp>
#include <popart_compiler/PopartEnums.hpp>

namespace poptorch {

namespace detail {

struct CompilerImpl {
public:
  friend Compiler;

  CompilerImpl() : op_builder(popart::Builder::create()), active_ipu(0) {}

  std::unique_ptr<popart::Builder> op_builder;

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

  popart::WeightsIO weight_callback;

  bool is_training;

  // Record each loss as it is used so we can make them inputs of the global
  // identity op.
  std::vector<popart::TensorId> losses;

  popart::SessionOptions popart_options;
  struct Options {
    bool profile;
    std::uint64_t steps;
    PopartAnchorTypes anchor_mode;
    std::uint64_t anchor_return_period;
    bool ipu_model;
    std::uint64_t ipu_version;
    std::uint64_t ipu_id;
    popart::DeviceConnectionType connection_type;
    popart::SyncPattern sync_pattern;
  };

  // List of options which have been explicitely set by the user.
  std::set<std::string> options_set;

  Options options;

  // We add operations using a state based system so the user would set the
  // active IPU and all subsequent operations will be added to that IPU until
  // stopped.
  std::uint64_t active_ipu;

  std::unordered_set<std::uint64_t> used_ipus;

  // General helpers.

  // Inserts memory into the list of tensors being output by the model.
  void addMemoryToOutput(poptorch::TensorId id, void *ptr,
                         std::unique_ptr<popart::IArray> &&memory);

  // Domain helpers
  popart::TensorId reshape(const std::vector<popart::TensorId> &inputs,
                           const std::vector<int64_t> &shape);

  popart::TensorId cast(const std::vector<popart::TensorId> &inputs,
                        const std::string &type);

  popart::TensorId intConstant(const std::vector<popart::TensorId> &inputs,
                               const std::vector<int32_t> &data,
                               const std::vector<int64_t> &shape);

  popart::TensorId floatConstant(const std::vector<popart::TensorId> &inputs,
                                 const std::vector<double> &data,
                                 const std::vector<int64_t> &shape);

  popart::TensorId addNotInPlace(const std::vector<popart::TensorId> &in);

  void updateUseModelConfig();
  std::string checkSystemConfig();
};

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
  } else if (const char *env_use_model = std::getenv("POPTORCH_IPU_MODEL")) {
    // As a fallback the model can be enabled by the POPTORCH_IPU_MODEL
    // environment variable.
    options.ipu_model = std::stoi(env_use_model) != 0;
    logging::info("From POPTORCH_IPU_MODEL environment variable: Ipu model: {}",
                  options.ipu_model ? "Enabled" : "Disabled");
  } else {
    options.ipu_model = false;
  }
}

void CompilerImpl::addMemoryToOutput(poptorch::TensorId id, void *ptr,
                                     std::unique_ptr<popart::IArray> &&memory) {
  memory_manager.push_back(std::move(memory));

  popart::TensorId popart_id = ids[id];
  if (!popart_outgoing.insert({popart_id, *memory_manager.back().get()})
           .second) {
    // Insertion in the map failed because there is already a pointer associated
    // with that id.
    outgoing_duplicates[popart_id].push_back(ptr);
  }
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
  bool_options["profile"] = [&](bool value) {
    poptorch_options.profile = value;
  };
  bool_options["use_model"] = [&](bool value) {
    poptorch_options.ipu_model = value;
  };
  bool_options["constant_weights"] = [&](bool value) {
    popart_options.constantWeights = value;
  };
  bool_options["enable_pipelining"] = [&](bool value) {
    popart_options.enablePipelining = value;
  };

  uint64_options["device_iterations"] = [&](std::uint64_t value) {
    poptorch_options.steps = value;
  };
  uint64_options["ipu_version"] = [&](std::uint64_t value) {
    poptorch_options.ipu_version = value;
  };
  uint64_options["ipu_id"] = [&](std::uint64_t value) {
    poptorch_options.ipu_id = value;
  };
  uint64_options["gradient_accumulation"] = [&](std::uint64_t value) {
    popart_options.accumulationFactor = value;
  };
  uint64_options["anchor_return_period"] = [&](std::uint64_t value) {
    poptorch_options.anchor_return_period = value;
  };
  uint64_options["replication_factor"] = [&](std::uint64_t value) {
    popart_options.replicatedGraphCount = value;
  };

  uint64_options["anchor_mode"] = [&](std::uint64_t value) {
    ERROR_ON_MSG(value >= static_cast<std::uint64_t>(PopartAnchorTypes::N),
                 "Value for PopartAnchorTypes out of range");
    poptorch_options.anchor_mode = static_cast<PopartAnchorTypes>(value);
  };

  uint64_options["connection_type"] = [&](std::uint64_t value) {
    ERROR_ON_MSG(
        value > static_cast<std::uint64_t>(popart::DeviceConnectionType::Never),
        "Value for DeviceConnectionType out of range");
    poptorch_options.connection_type =
        static_cast<popart::DeviceConnectionType>(value);
  };

  uint64_options["sync_pattern"] = [&](std::uint64_t value) {
    ERROR_ON_MSG(value >
                     static_cast<std::uint64_t>(popart::SyncPattern::PingPong),
                 "Value for SyncPattern out of range");
    poptorch_options.sync_pattern = static_cast<popart::SyncPattern>(value);
  };

  string_options["log_dir"] = [&](const std::string &value) {
    popart_options.logDir = value;
  };

  string_options["logDir"] = [&](const std::string &log_dir) {
    UNUSED(log_dir);
    logging::warn(
        "Ignoring call to poptorch.Options.Popart.set(\"logDir\",...): use "
        "poptorch.Options.logDir() instead");
  };

  container_options["dotChecks"] =
      [&](const std::pair<std::string, std::string> &p) {
        std::uint64_t value = std::stoul(p.first);
        ERROR_ON_MSG(value >= static_cast<std::uint64_t>(popart::DotCheck::N),
                     "Value for DotCheck out of range");
        popart_options.dotChecks.insert(static_cast<popart::DotCheck>(value));
      };

  container_options["hardwareInstrumentations"] =
      [&](const std::pair<std::string, std::string> &p) {
        std::uint64_t value = std::stoul(p.first);
        ERROR_ON_MSG(value >=
                         static_cast<std::uint64_t>(popart::Instrumentation::N),
                     "Value for Instrumentation out of range");
        popart_options.hardwareInstrumentations.insert(
            static_cast<popart::Instrumentation>(value));
      };

  container_options["customCodelets"] =
      [&](const std::pair<std::string, std::string> &p) {
        popart_options.customCodelets.push_back(p.first);
      };

  container_options["engineOptions"] =
      [&](const std::pair<std::string, std::string> &p) {
        popart_options.engineOptions.emplace(p);
      };

  container_options["reportOptions"] =
      [&](const std::pair<std::string, std::string> &p) {
        popart_options.reportOptions.emplace(p);
      };

  container_options["convolutionOptions"] =
      [&](const std::pair<std::string, std::string> &p) {
        popart_options.convolutionOptions.emplace(p);
      };

#define ADD_POPART_ENUM_OPTION(name, EnumType)                                 \
  uint64_options[#name] = [&](std::uint64_t value) {                           \
    ERROR_ON_MSG(value >= static_cast<std::uint64_t>(popart::EnumType::N),     \
                 "Value for " << #EnumType << " out of range");                \
    popart_options.name = static_cast<popart::EnumType>(value);                \
  }

#define ADD_POPART_BOOL_OPTION(name)                                           \
  bool_options[#name] = [&](bool value) { popart_options.name = value; }
#define ADD_POPART_UINT64_OPTION(name)                                         \
  uint64_options[#name] = [&](std::uint64_t value) {                           \
    popart_options.name = value;                                               \
  }

#define ADD_POPART_DOUBLE_OPTION(name)                                         \
  double_options[#name] = [&](double value) { popart_options.name = value; }

#define ADD_POPART_STRING_OPTION(name)                                         \
  string_options[#name] = [&](const std::string &value) {                      \
    popart_options.name = value;                                               \
  }

  ADD_POPART_ENUM_OPTION(autoRecomputation, RecomputationType);
  ADD_POPART_ENUM_OPTION(mergeVarUpdate, MergeVarUpdateType);
  ADD_POPART_ENUM_OPTION(virtualGraphMode, VirtualGraphMode);
  ADD_POPART_ENUM_OPTION(syntheticDataMode, SyntheticDataMode);

  ADD_POPART_STRING_OPTION(cachePath);
  ADD_POPART_STRING_OPTION(partialsTypeMatMuls);
  ADD_POPART_STRING_OPTION(customCodeletCompileFlags);
  ADD_POPART_STRING_OPTION(serializedPoprithmsAnnealGraphsDir);
  ADD_POPART_STRING_OPTION(kahnTieBreaker);
  ADD_POPART_STRING_OPTION(ipuSystemType);

  ADD_POPART_UINT64_OPTION(firstDotOp);
  ADD_POPART_UINT64_OPTION(finalDotOp);
  ADD_POPART_UINT64_OPTION(pingPongPhases);
  ADD_POPART_UINT64_OPTION(numIOTiles);
  ADD_POPART_UINT64_OPTION(batchSerializationFactor);
  ADD_POPART_UINT64_OPTION(mergeVarUpdateMemThreshold);
  ADD_POPART_UINT64_OPTION(looseThresholdAtPeak);
  ADD_POPART_UINT64_OPTION(accumulationFactor);
  ADD_POPART_UINT64_OPTION(swapLimitScheduler);
  ADD_POPART_UINT64_OPTION(globalReplicationFactor);
  ADD_POPART_UINT64_OPTION(globalReplicaOffset);
  ADD_POPART_UINT64_OPTION(replicatedWeightShardingMinNumElements);

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
  ADD_POPART_BOOL_OPTION(replicatedWeightSharding);
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
  auto ai_onnx = op_builder->aiOnnxOpset9();
  return op_builder->reshape_const(ai_onnx, inputs, shape);
}

popart::TensorId CompilerImpl::cast(const std::vector<popart::TensorId> &inputs,
                                    const std::string &type) {
  auto ai_onnx = op_builder->aiOnnxOpset9();
  return ai_onnx.cast(inputs, type);
}

popart::TensorId
CompilerImpl::intConstant(const std::vector<popart::TensorId> &inputs,
                          const std::vector<int32_t> &data,
                          const std::vector<int64_t> &shape) {
  UNUSED(inputs);
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{"INT32", shape};

  std::int64_t total_size = std::accumulate(shape.begin(), shape.end(), 1,
                                            std::multiplies<std::int64_t>());
  std::vector<int32_t> broadcasted_data(total_size);

  // Create the inital data for the variable.
  popart::ConstVoidData the_data;

  if (data.size() == 1 && total_size != 1) {
    std::for_each(broadcasted_data.begin(), broadcasted_data.end(),
                  [&data](std::int32_t &i) { i = data[0]; });

    the_data.data = broadcasted_data.data();
    the_data.info = info;
  } else {
    the_data.data = data.data();
    the_data.info = info;
  }

  auto ai_onnx = op_builder->aiOnnxOpset9();
  return ai_onnx.constant(the_data);
}

popart::TensorId
CompilerImpl::floatConstant(const std::vector<popart::TensorId> &inputs,
                            const std::vector<double> &data,
                            const std::vector<int64_t> &shape) {
  UNUSED(inputs);
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{"FLOAT", shape};

  std::int64_t total_size = std::accumulate(shape.begin(), shape.end(), 1,
                                            std::multiplies<std::int64_t>());
  std::vector<float> broadcasted_data(total_size);

  // Create the inital data for the variable.
  popart::ConstVoidData the_data;

  if (data.size() == 1 && total_size != 1) {
    std::for_each(broadcasted_data.begin(), broadcasted_data.end(),
                  [&data](float &i) { i = data[0]; });

    the_data.data = broadcasted_data.data();
    the_data.info = info;
  } else {
    int counter = 0;
    std::for_each(broadcasted_data.begin(), broadcasted_data.end(),
                  [&](float &i) { i = data[counter++]; });

    the_data.data = broadcasted_data.data();
    the_data.info = info;
  }

  auto ai_onnx = op_builder->aiOnnxOpset9();
  return ai_onnx.constant(the_data);
}

} // namespace detail

bool ipuHardwareIsAvailable(std::uint64_t num_ipus) {
  return !popart::DeviceManager::createDeviceManager()
              .enumerateDevices(popart::SyncPattern::Full, num_ipus)
              .empty();
}

// Variadic output case. For now we will add all outputs to the graph and
// allocate them on the same IPU but we will only return one. This means only
// one output can be used by user IR (but can still be used by the backed via
// transformations).
template <typename T> struct HandleOutput {
  poptorch::TensorId operator()(T &in, bool loss, detail::CompilerImpl *_impl) {
    std::set<popart::TensorId> ids;

    for (const popart::TensorId &id : in) {
      ids.insert(id);
      _impl->ids.push_back(id);

      if (loss) {
        _impl->losses.push_back(id);
      }
    }

    _impl->op_builder->virtualGraph(ids, _impl->active_ipu);
    _impl->used_ipus.insert(_impl->active_ipu);

    // Return the first added tensor as the sole return of this IR op.
    return _impl->ids.size() - in.size();
  }
};

// Single tensor output case
template <> struct HandleOutput<popart::TensorId> {
  poptorch::TensorId operator()(const popart::TensorId &in, bool loss,
                                detail::CompilerImpl *_impl) {
    _impl->op_builder->virtualGraph(in, _impl->active_ipu);
    _impl->used_ipus.insert(_impl->active_ipu);
    _impl->ids.push_back(in);

    if (loss) {
      _impl->losses.push_back(in);
    }

    return _impl->ids.size() - 1;
  }
};

namespace detail {

popart::TensorId
CompilerImpl::addNotInPlace(const std::vector<popart::TensorId> &in) {
  auto ai_onnx = op_builder->aiOnnxOpset9();
  popart::TensorId output = ai_onnx.add(in);
  op_builder->setInplacePreferences(
      output, {{"AddLhsInplace", -1}, {"AddRhsInplace", -1}});
  return output;
}

} // namespace detail

poptorch::TensorId
Compiler::addInputTensor(const char *type,
                         const std::vector<std::int64_t> &dims) {
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{type, dims};
  _impl->ids.push_back(_impl->op_builder->addInputTensor(info));
  return _impl->ids.size() - 1;
}

std::vector<std::int32_t> int64ToInt32(const std::vector<std::int64_t> &in) {
  std::vector<std::int32_t> x;

  for (std::int64_t i : in) {
    // If i is less than the int32 smallest value or greater than its biggest,
    // throw overflow error.
    bool overflow =
        i > static_cast<std::int64_t>(
                std::numeric_limits<std::int32_t>::max()) ||
        i < static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min());

    ERROR_ON_MSG(overflow, "Int 64 overflowed during poptorch compilation.");
    x.push_back(i);
  }

  return x;
}

// A whitelist of supported loss operations. Popart needs to know which
// operations are losses so they can be marked by the session.
static bool IsLoss(const std::string &operation) {
  return operation == "popart::l1loss" || operation == "popart::nllloss" ||
         operation == "popart::identityloss";
}

#define INT_VEC std::vector<std::int64_t>
#define FLOAT_VEC std::vector<double>
#define FLOAT float
#define INT std::int64_t
#define BOOL bool
#define STRING const char *
#define NONE
#define ARG(Type, Name) , Type Name
#define BODY_ARG(Name) , Name

// Create a function decl with the given call and arguments.
#define OP_DECL(ns, funcName, function, onnxImpl, Args, BodyArgs)              \
  poptorch::TensorId Compiler::function(                                       \
      const std::vector<poptorch::TensorId> &inputs Args) {                    \
    auto AiOnnxOpset9 = _impl->op_builder->aiOnnxOpset9();                     \
    auto AiGraphcoreOpset1 = _impl->op_builder->aiGraphcoreOpset1();           \
    const bool isLoss = IsLoss(#ns "::" #funcName);                            \
    std::vector<popart::TensorId> ins;                                         \
    std::transform(                                                            \
        inputs.begin(), inputs.end(), std::back_inserter(ins),                 \
        [&](poptorch::TensorId index) { return _impl->ids[index]; });          \
    auto output = onnxImpl(ins BodyArgs);                                      \
    return HandleOutput<decltype(output)>{}(output, isLoss, _impl.get());      \
  }

#include "popart_compiler/SupportedOperations.inc.hpp"

#undef OP_DECL
#undef BODY_ARG
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
      _impl->op_builder->addInitializedInputTensor(the_data, name));

  popart::TensorId id = _impl->ids[_impl->ids.size() - 1];

  popart::MutableVoidData mutable_data;
  mutable_data.data = data;
  mutable_data.info = info;

  _impl->weight_callback.insert(id, mutable_data);

  return _impl->ids.size() - 1;
}

void Compiler::addOutputTensor(poptorch::TensorId output) {
  _impl->outputs.push_back(_impl->ids[output]);
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
  assertTensorIs(PopartTypes::FLOAT, id,
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
  assertTensorIs(PopartTypes::INT32, id,
                 static_cast<const char *>(__PRETTY_FUNCTION__));

  // Popart wrapper around the tensor pointer.
  _impl->memory_manager.push_back(
      std::make_unique<popart::NDArrayWrapper<std::int32_t>>(ptr, dims));
  _impl->popart_incoming.insert(
      {_impl->ids[id], *_impl->memory_manager.back().get()});
}

void Compiler::setUpInputOp(poptorch::TensorId id, std::int16_t *ptr,
                            const std::vector<std::int64_t> &dims,
                            bool float16) {
  if (float16) {
    assertTensorIs(PopartTypes::FLOAT16, id,
                   static_cast<const char *>(__PRETTY_FUNCTION__));

  } else {
    assertTensorIs(PopartTypes::INT16, id,
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

void Compiler::initSession(const Optimizer &opt) {
  popart::SessionOptions &options = _impl->popart_options;
  _impl->updateUseModelConfig();

  std::shared_ptr<popart::DeviceInfo> device;
  if (_impl->options.ipu_model) {
    ERROR_ON_MSG(_impl->options.connection_type ==
                     popart::DeviceConnectionType::Never,
                 "ConnectionType.Never / poptorch.Options.useOfflineIpuTarget "
                 "not supported for the IPU model");
    device = popart::DeviceManager::createDeviceManager().createCpuDevice();
    logging::debug("Instantiated Cpu device, running on IPU model.");
  } else {
    const std::uint64_t num_ipus =
        _impl->used_ipus.size() * options.replicatedGraphCount;
    if (_impl->options.connection_type == popart::DeviceConnectionType::Never) {
      // Offline compilation path: create an offline device regardless of what's
      // present on the system.
      ERROR_ON_MSG(_impl->options_set.count("ipu_id"),
                   "Offline compilation targeting a specific id not supported");
      std::map<std::string, std::string> device_options;
      device_options["numIPUs"] = std::to_string(num_ipus);
      device_options["ipu_version"] =
          "ipu" + std::to_string(_impl->options.ipu_version);
      device_options["sync_pattern"] =
          popart::syncPatternToString(_impl->options.sync_pattern);
      device =
          popart::DeviceManager::createDeviceManager().createOfflineIPUDevice(
              device_options);
      ERROR_ON_MSG(!device, "Failed to create offline IPU device");
    } else {
      // Regular IPU hardware target
      if (!_impl->options_set.count("ipu_id")) {
        device =
            popart::DeviceManager::createDeviceManager().acquireAvailableDevice(
                num_ipus, 0, _impl->options.sync_pattern,
                _impl->options.connection_type);
        ERROR_ON_MSG(!device, "Failed to acquire "
                                  << num_ipus << " IPU(s)"
                                  << _impl->checkSystemConfig());
        logging::debug("Acquired IPU device, running on device.");
      } else {
        device = popart::DeviceManager::createDeviceManager().acquireDeviceById(
            _impl->options.ipu_id, _impl->options.sync_pattern,
            _impl->options.connection_type);
        ERROR_ON_MSG(!device, "Failed to acquire device Id "
                                  << _impl->options.ipu_id
                                  << _impl->checkSystemConfig());
        ERROR_ON_MSG(static_cast<std::uint64_t>(device->getNumIpus()) !=
                         num_ipus,
                     "Expected replication factor * used IPUs = "
                         << _impl->used_ipus.size() << " * "
                         << options.replicatedGraphCount << " = " << num_ipus
                         << " device Ids but the user provided "
                         << device->getNumIpus());
        logging::debug("Acquired IPU device with id {}, running on device.",
                       _impl->options.ipu_id);
      }
    }
  }

  // If Pipelining wasn't set: enable it if more than 1 IPU is used.
  if (!_impl->options_set.count("enablePipelining")) {
    options.enablePipelining = _impl->used_ipus.size() > 1;
  }

  bool enable_replicated_graphs = options.replicatedGraphCount != 1;
  if (_impl->options_set.count("enable_replicated_graphs") &&
      options.enableReplicatedGraphs != enable_replicated_graphs) {
    logging::warn("enable_replicated_graphs forced by the user to {}",
                  options.enableReplicatedGraphs);
  } else {
    options.enableReplicatedGraphs = enable_replicated_graphs;
  }

  logging::info("Popart replication enabled: {} with factor set to {}",
                options.enableReplicatedGraphs, options.replicatedGraphCount);

  // Disable constant_weights by default: causes problems with Popart
  const bool constant_weights = false;
  if (_impl->options_set.count("constant_weights") &&
      options.constantWeights != constant_weights) {
    logging::warn("constant_weights forced by the user to {}",
                  options.constantWeights);
  } else {
    options.constantWeights = constant_weights;
  }

  if (_impl->used_ipus.size() > 1) {
    if (!options.enablePipelining) {
      logging::warn("Using {} IPUs but "
                    "poptorch.Options.enablePipelining() is False",
                    _impl->used_ipus.size());
    }
    if (_impl->options_set.count("virtualGraphMode") &&
        options.virtualGraphMode != popart::VirtualGraphMode::Manual) {
      logging::warn("virtualGraphMode forced by the user to {} ",
                    popart::toString(options.virtualGraphMode));
    } else {
      options.virtualGraphMode = popart::VirtualGraphMode::Manual;
    }
  }

  bool enable_gradient_accumulation = options.accumulationFactor > 1;
  if (_impl->options_set.count("enable_gradient_accumulation") &&
      !options.enableGradientAccumulation) {
    logging::warn("enable_gradient_accumulation forced by the user to {}",
                  options.enableGradientAccumulation);
  } else {
    options.enableGradientAccumulation = enable_gradient_accumulation;
  }

  // Create the anchors, these are used to copy to the host.
  auto data_flow = popart::DataFlow(_impl->options.steps, _impl->anchors);

  // Create the popart session object to actually run the graph.
  if (!_impl->is_training) {
    // Create an inference session.
    _impl->session = popart::InferenceSession::createFromOnnxModel(
        _impl->op_builder->getModelProto(), data_flow, device, {}, options,
        popart::PatternsLevel::Default);
  } else {
    logging::debug(
        "Adding initial graph optimizer SGD with parameters:: Learning rate "
        "{}, weight decay {}, Momentum {}, Dampening {}",
        opt.learning_rate.first, opt.weight_decay.first, opt.momentum.first,
        opt.dampening.first);

    // Create the optimizer from user provided parameters.
    auto optimizer =
        popart::SGD(opt.learning_rate, opt.weight_decay, opt.momentum,
                    opt.dampening, {1.0f, true}, // Velocity scaling, off.
                    {1.0f, true});               // Loss scaling, off.

    // set a global identity loss that all other losses derive from.
    popart::TensorId loss_root =
        _impl->op_builder->aiGraphcoreOpset1().identityloss(_impl->losses);
    _impl->op_builder->virtualGraph(loss_root, _impl->active_ipu);

    // Transform nodes which have training/inference variants. I.E BatchNorm.
    popart::GraphTransformer transformer{_impl->op_builder->getModelProto()};
    transformer.prepareNodesForTraining();

    // Create the training session.
    _impl->session = popart::TrainingSession::createFromOnnxModel(
        transformer.getModelProto(), data_flow, loss_root, optimizer, device,
        {}, options, popart::PatternsLevel::Default);
  }

  logging::trace(
      "Popart serialised IR:\n{}",
      _impl->session->serializeIr(popart::IrSerializationFormat::JSON));

  // Poplar compilation.
  try {
    logging::trace("Begining Poplar compilation.");
    _impl->session->prepareDevice();
    logging::trace("Finished Poplar compilation.");
  } catch (popart::memory_allocation_err &e) {
    std::ofstream stream;
    stream.open("OOMReport.json");
    stream << e.getGraphReport(true);
    stream.close();

    std::rethrow_exception(std::current_exception());
  }

  if (_impl->options.profile) {
    std::ofstream stream;
    stream.open("GraphReport.json");
    stream << _impl->session->getGraphReport();
    stream.close();
  }

  // Write the weights immediately after compilation to the IPU.
  copyWeightsToDevice();
}

// Write the weights into IPU memory from the pytorch tensor buffers in the
// model.
void Compiler::copyWeightsToDevice() {
  logging::info("Writing weights from host to IPU memory.");
  _impl->session->writeWeights(_impl->weight_callback);
  _impl->session->weightsFromHost();
}

// Read the weights from IPU memory into the pytorch tensor buffers.
void Compiler::copyWeightsToHost() {
  logging::info("Writing weights from IPU to host.");
  _impl->session->weightsToHost();
  _impl->session->readWeights(_impl->weight_callback);
}

void Compiler::run(const Optimizer &optimizer) {
  if (optimizer.type != OptimizerType::NONE && _impl->is_training) {
    // Convert the map from the user into a popart SGD class.
    auto new_optimizer = popart::SGD(
        optimizer.learning_rate, optimizer.weight_decay, optimizer.momentum,
        optimizer.dampening, {1.0f, true}, // Velocity scaling, off.
        {1.0f, true});                     // Loss scaling, off.

    // Print to debug the new optimizer.
    logging::debug(
        "Updating graph optimizer SGD with parameters: Learning rate "
        "{}, weight decay {}, Momentum {}, Dampening {}",
        optimizer.learning_rate.first, optimizer.weight_decay.first,
        optimizer.momentum.first, optimizer.dampening.first);

    // Update the popart graph/poplar executable with the new optimizer.
    popart::TrainingSession &session =
        dynamic_cast<popart::TrainingSession &>(*_impl->session);
    session.updateOptimizerFromHost(&new_optimizer);
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

poptorch::PopartTypes Compiler::getPopartType(poptorch::TensorId tensor) const {
  popart::TensorInfo info = _impl->session->getInfo(_impl->ids[tensor]);

  switch (info.dataType()) {
  case popart::DataType::UINT8: {
    return PopartTypes::UINT8;
  }
  case popart::DataType::INT8: {
    return PopartTypes::INT8;
  }
  case popart::DataType::UINT16: {
    return PopartTypes::UINT16;
  }
  case popart::DataType::INT16: {
    return PopartTypes::INT16;
  }
  case popart::DataType::INT32: {
    return PopartTypes::INT32;
  }
  case popart::DataType::INT64: {
    return PopartTypes::INT64;
  }
  case popart::DataType::UINT32: {
    return PopartTypes::UINT32;
  }
  case popart::DataType::UINT64: {
    return PopartTypes::UINT64;
  }
  case popart::DataType::BOOL: {
    return PopartTypes::BOOL;
  }
  case popart::DataType::FLOAT: {
    return PopartTypes::FLOAT;
  }
  case popart::DataType::FLOAT16: {
    return PopartTypes::FLOAT16;
  }
  case popart::DataType::BFLOAT16: {
    return PopartTypes::BFLOAT16;
  }
  case popart::DataType::DOUBLE: {
    return PopartTypes::DOUBLE;
  }
  case popart::DataType::COMPLEX64: {
    return PopartTypes::COMPLEX64;
  }
  case popart::DataType::COMPLEX128: {
    return PopartTypes::COMPLEX128;
  }
  case popart::DataType::STRING: {
    return PopartTypes::STRING;
  }
  case popart::DataType::UNDEFINED: {
    return PopartTypes::UNDEFINED;
  }
  }

  ERROR("Unsupported popart type in return: " << info.data_type());
}

bool Compiler::tensorIdIsValid(poptorch::TensorId id) const {
  return id < _impl->ids.size();
}

std::vector<std::int64_t> Compiler::getSize(poptorch::TensorId id) const {
  popart::TensorInfo info = _impl->session->getInfo(_impl->ids[id]);

  return info.shape();
}

void Compiler::setActiveIpu(std::uint64_t id) { _impl->active_ipu = id; }

std::uint64_t Compiler::batchPerStep() const { return _impl->options.steps; }

std::uint64_t Compiler::popartBatchDim() const {
  return _impl->popart_options.replicatedGraphCount * _impl->options.steps *
         _impl->popart_options.accumulationFactor;
}

std::uint64_t Compiler::popartBatchDimForAnchor(poptorch::TensorId id) const {
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

void Compiler::assertTensorIs(const PopartTypes dataType,
                              const poptorch::TensorId &id,
                              const char *caller) const {
  PopartTypes actual_type;
  try {
    actual_type = getPopartType(id);
  } catch (const popart::error &) {
    // Rare case of input tensor never used, so not in IR
    return;
  }

  ERROR_ON_MSG(actual_type != dataType,
               "Incorrect type for tensor, " << id << " used in " << caller);
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

SessionOptions::~SessionOptions() = default;
} // namespace poptorch
