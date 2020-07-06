// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart_compiler/Compiler.hpp>

#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
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

namespace poptorch {

namespace detail {

struct CompilerImpl {
public:
  friend Compiler;

  CompilerImpl() : opBuilder(popart::Builder::create()), activeIpu(0) {}

  std::unique_ptr<popart::Builder> opBuilder;

  std::map<popart::TensorId, popart::AnchorReturnType> anchors;

  std::vector<popart::TensorId> ids;

  // Input tensors to the session.
  std::map<popart::TensorId, popart::IArray &> popartIncoming;

  // Output tensors for the session.
  std::map<popart::TensorId, popart::IArray &> popartOutgoing;
  std::map<popart::TensorId, std::vector<void *>> outgoingDuplicates;

  std::list<popart::TensorId> outputs;
  // Flat representation of the output shapes
  std::vector<OutputType> outputTypes;

  // A list to allocate our buffers in so they get released.
  std::list<std::unique_ptr<popart::IArray>> memoryManager;

  std::unique_ptr<popart::Session> session;

  popart::WeightsIO weightCallback;

  bool isTraining;

  // Record each loss as it is used so we can make them inputs of the global
  // identity op.
  std::vector<popart::TensorId> losses;

  popart::SessionOptions popartOptions;
  struct Options {
    bool profile;
    std::uint64_t steps;
    PopartAnchorTypes anchorMode;
    std::uint64_t anchorReturnPeriod;
  };

  // List of options which have been explicitely set by the user.
  std::set<std::string> optionsSet;

  Options options;

  // We add operations using a state based system so the user would set the
  // active IPU and all subsequent operations will be added to that IPU until
  // stopped.
  std::uint64_t activeIpu;

  std::unordered_set<std::uint64_t> usedIpus;

  // General helpers.

  // Inserts memory into the list of tensors being output by the model.
  void AddMemoryToOutput(poptorch::TensorId id, void *ptr,
                         std::unique_ptr<popart::IArray> &&memory);

  // Domain helpers
  popart::TensorId reshape(const std::vector<popart::TensorId> &inputs,
                           const std::vector<int64_t> &shape);

  popart::TensorId intConstant(const std::vector<popart::TensorId> &inputs,
                               const std::vector<int32_t> &data,
                               const std::vector<int64_t> &shape);

  popart::TensorId floatConstant(const std::vector<popart::TensorId> &inputs,
                                 const std::vector<double> &data,
                                 const std::vector<int64_t> &shape);

  popart::TensorId cast(const std::vector<popart::TensorId> &inputs,
                        const std::string &type);

  popart::TensorId addNotInPlace(const std::vector<popart::TensorId> &in);
};

void CompilerImpl::AddMemoryToOutput(poptorch::TensorId id, void *ptr,
                                     std::unique_ptr<popart::IArray> &&memory) {
  memoryManager.push_back(std::move(memory));

  popart::TensorId popartId = ids[id];
  if (!popartOutgoing.insert({popartId, *memoryManager.back().get()}).second) {
    // Insertion in the map failed because there is already a pointer associated
    // with that id.
    outgoingDuplicates[popartId].push_back(ptr);
  }
}

struct SessionOptionsImpl {
  SessionOptionsImpl();

  std::map<std::string, std::function<void(bool)>> boolOptions;
  std::map<std::string, std::function<void(std::uint64_t)>> uint64Options;
  std::map<std::string, std::function<void(std::string)>> stringOptions;
  std::map<std::string, std::function<void(double)>> doubleOptions;
  std::map<std::string,
           std::function<void(std::pair<std::string, std::string>)>>
      containerOptions;
  std::set<std::string> optionsSet;

  popart::SessionOptions popartOptions;
  CompilerImpl::Options poptorchOptions;

  template <typename ValueType>
  void Set(const std::string &key, ValueType value,
           std::map<std::string, std::function<void(ValueType)>> &options,
           const std::string &typeStr) {
    auto it = options.find(key);
    ERROR_ON_MSG(it == options.end(),
                 "Unknown " << typeStr << " option " << key);
    it->second(value);
    optionsSet.insert(key);
  }
};

SessionOptionsImpl::SessionOptionsImpl() : popartOptions(), poptorchOptions() {
  // The keys must match the name and type of the attributes of SessionOptions
  // in python/__init__.py
  boolOptions["profile"] = [&](bool value) { poptorchOptions.profile = value; };
  boolOptions["constant_weights"] = [&](bool value) {
    popartOptions.constantWeights = value;
  };
  boolOptions["enable_pipelining"] = [&](bool value) {
    popartOptions.enablePipelining = value;
  };

  uint64Options["device_iterations"] = [&](std::uint64_t value) {
    poptorchOptions.steps = value;
  };
  uint64Options["gradient_accumulation"] = [&](std::uint64_t value) {
    popartOptions.accumulationFactor = value;
  };
  uint64Options["anchor_return_period"] = [&](std::uint64_t value) {
    poptorchOptions.anchorReturnPeriod = value;
  };
  uint64Options["replication_factor"] = [&](std::uint64_t value) {
    popartOptions.replicatedGraphCount = value;
  };

  uint64Options["anchor_mode"] = [&](std::uint64_t value) {
    ERROR_ON_MSG(value >= static_cast<std::uint64_t>(PopartAnchorTypes::N),
                 "Value for PopartAnchorTypes out of range");
    poptorchOptions.anchorMode = static_cast<PopartAnchorTypes>(value);
  };

  stringOptions["log_dir"] = [&](std::string value) {
    popartOptions.logDir = value;
  };

  stringOptions["logDir"] = [&](std::string) {
    logging::warn(
        "Ignoring call to poptorch.Options.Popart.Set(\"logDir\",...): use "
        "poptorch.Options.logDir() instead");
  };

  containerOptions["dotChecks"] = [&](std::pair<std::string, std::string> p) {
    std::uint64_t value = std::stoul(p.first);
    ERROR_ON_MSG(value >= static_cast<std::uint64_t>(popart::DotCheck::N),
                 "Value for DotCheck out of range");
    popartOptions.dotChecks.insert(static_cast<popart::DotCheck>(value));
  };

  containerOptions["hardwareInstrumentations"] =
      [&](std::pair<std::string, std::string> p) {
        std::uint64_t value = std::stoul(p.first);
        ERROR_ON_MSG(value >=
                         static_cast<std::uint64_t>(popart::Instrumentation::N),
                     "Value for Instrumentation out of range");
        popartOptions.hardwareInstrumentations.insert(
            static_cast<popart::Instrumentation>(value));
      };

  containerOptions["customCodelets"] =
      [&](std::pair<std::string, std::string> p) {
        popartOptions.customCodelets.push_back(p.first);
      };

  containerOptions["engineOptions"] =
      [&](std::pair<std::string, std::string> p) {
        popartOptions.engineOptions.emplace(p);
      };

  containerOptions["reportOptions"] =
      [&](std::pair<std::string, std::string> p) {
        popartOptions.reportOptions.emplace(p);
      };

  containerOptions["convolutionOptions"] =
      [&](std::pair<std::string, std::string> p) {
        popartOptions.convolutionOptions.emplace(p);
      };

#define ADD_POPART_ENUM_OPTION(name, EnumType)                                 \
  uint64Options[#name] = [&](std::uint64_t value) {                            \
    ERROR_ON_MSG(value >= static_cast<std::uint64_t>(popart::EnumType::N),     \
                 "Value for " << #EnumType << " out of range");                \
    popartOptions.name = static_cast<popart::EnumType>(value);                 \
  }

#define ADD_POPART_BOOL_OPTION(name)                                           \
  boolOptions[#name] = [&](bool value) { popartOptions.name = value; }
#define ADD_POPART_UINT64_OPTION(name)                                         \
  uint64Options[#name] = [&](std::uint64_t value) {                            \
    popartOptions.name = value;                                                \
  }

#define ADD_POPART_DOUBLE_OPTION(name)                                         \
  doubleOptions[#name] = [&](double value) { popartOptions.name = value; }

#define ADD_POPART_STRING_OPTION(name)                                         \
  stringOptions[#name] = [&](std::string value) { popartOptions.name = value; }

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
  auto aiOnnx = opBuilder->aiOnnxOpset9();
  return opBuilder->reshape_const(aiOnnx, inputs, shape);
}

popart::TensorId
CompilerImpl::intConstant(const std::vector<popart::TensorId> &inputs,
                          const std::vector<int32_t> &data,
                          const std::vector<int64_t> &shape) {
  UNUSED(inputs);
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{"INT32", shape};

  std::int64_t totalSize = std::accumulate(shape.begin(), shape.end(), 1,
                                           std::multiplies<std::int64_t>());
  std::vector<int32_t> broadcastedData(totalSize);

  // Create the inital data for the variable.
  popart::ConstVoidData theData;

  if (data.size() == 1 && totalSize != 1) {
    std::for_each(broadcastedData.begin(), broadcastedData.end(),
                  [&data](std::int32_t &i) { i = data[0]; });

    theData.data = broadcastedData.data();
    theData.info = info;
  } else {
    theData.data = data.data();
    theData.info = info;
  }

  auto aiOnnx = opBuilder->aiOnnxOpset9();
  return aiOnnx.constant(theData);
}

popart::TensorId CompilerImpl::cast(const std::vector<popart::TensorId> &inputs,
                                    const std::string &type) {
  auto aiOnnx = opBuilder->aiOnnxOpset9();
  return aiOnnx.cast(inputs, type);
}

popart::TensorId
CompilerImpl::floatConstant(const std::vector<popart::TensorId> &inputs,
                            const std::vector<double> &data,
                            const std::vector<int64_t> &shape) {
  UNUSED(inputs);
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{"FLOAT", shape};

  std::int64_t totalSize = std::accumulate(shape.begin(), shape.end(), 1,
                                           std::multiplies<std::int64_t>());
  std::vector<float> broadcastedData(totalSize);

  // Create the inital data for the variable.
  popart::ConstVoidData theData;

  if (data.size() == 1 && totalSize != 1) {
    std::for_each(broadcastedData.begin(), broadcastedData.end(),
                  [&data](float &i) { i = data[0]; });

    theData.data = broadcastedData.data();
    theData.info = info;
  } else {
    int counter = 0;
    std::for_each(broadcastedData.begin(), broadcastedData.end(),
                  [&](float &i) { i = data[counter++]; });

    theData.data = broadcastedData.data();
    theData.info = info;
  }

  auto aiOnnx = opBuilder->aiOnnxOpset9();
  return aiOnnx.constant(theData);
}

} // namespace detail

// Variadic output case. For now we will add all outputs to the graph and
// allocate them on the same IPU but we will only return one. This means only
// one output can be used by user IR (but can still be used by the backed via
// transformations).
template <typename T> struct HandleOutput {
  poptorch::TensorId operator()(T &in, bool loss, detail::CompilerImpl *impl) {
    std::set<popart::TensorId> ids;

    for (popart::TensorId id : in) {
      ids.insert(id);
      impl->ids.push_back(id);

      if (loss) {
        impl->losses.push_back(id);
      }
    }

    impl->opBuilder->virtualGraph(ids, impl->activeIpu);
    impl->usedIpus.insert(impl->activeIpu);

    // Return the first added tensor as the sole return of this IR op.
    return impl->ids.size() - in.size();
  }
};

// Single tensor output case
template <> struct HandleOutput<popart::TensorId> {
  poptorch::TensorId operator()(popart::TensorId in, bool loss,
                                detail::CompilerImpl *impl) {
    impl->opBuilder->virtualGraph(in, impl->activeIpu);
    impl->usedIpus.insert(impl->activeIpu);
    impl->ids.push_back(in);

    if (loss) {
      impl->losses.push_back(in);
    }

    return impl->ids.size() - 1;
  }
};

namespace detail {

popart::TensorId
CompilerImpl::addNotInPlace(const std::vector<popart::TensorId> &in) {
  auto AiOnnxOpset9 = opBuilder->aiOnnxOpset9();
  popart::TensorId output = AiOnnxOpset9.add(in);
  opBuilder->setInplacePreferences(
      output, {{"AddLhsInplace", -1}, {"AddRhsInplace", -1}});
  return output;
}

} // namespace detail

poptorch::TensorId
Compiler::AddInputTensor(const char *string,
                         const std::vector<std::int64_t> &dims) {
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{string, dims};
  impl->ids.push_back(impl->opBuilder->addInputTensor(info));
  return impl->ids.size() - 1;
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
  if (operation == "popart::l1loss" || operation == "popart::nllloss" ||
      operation == "popart::identityloss") {
    return true;
  }

  return false;
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
    auto AiOnnxOpset9 = impl->opBuilder->aiOnnxOpset9();                       \
    auto AiGraphcoreOpset1 = impl->opBuilder->aiGraphcoreOpset1();             \
    const bool isLoss = IsLoss(#ns "::" #funcName);                            \
    std::vector<popart::TensorId> ins;                                         \
    std::transform(                                                            \
        inputs.begin(), inputs.end(), std::back_inserter(ins),                 \
        [&](poptorch::TensorId index) { return impl->ids[index]; });           \
    auto output = onnxImpl(ins BodyArgs);                                      \
    return HandleOutput<decltype(output)>{}(output, isLoss, impl.get());       \
  }

#include "popart_compiler/SupportedOperations.inc.h"
#undef BODY_ARG
#undef OP_DECL
#undef ARG
#undef NONE
#undef INT_VEC
#undef FLOAT_VEC
#undef FLOAT
#undef INT
#undef BOOL
#undef STRING

poptorch::TensorId
Compiler::AddInitializedInputTensor(const char *name, const char *type,
                                    const std::vector<std::int64_t> &dims,
                                    void *data) {
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{type, dims};

  // Create the inital data for the variable.
  popart::ConstVoidData theData;
  theData.data = data;
  theData.info = info;

  impl->ids.push_back(
      impl->opBuilder->addInitializedInputTensor(theData, name));

  popart::TensorId id = impl->ids[impl->ids.size() - 1];

  popart::MutableVoidData mutableData;
  mutableData.data = data;
  mutableData.info = info;

  impl->weightCallback.insert(id, mutableData);

  return impl->ids.size() - 1;
}

void Compiler::AddOutputTensor(poptorch::TensorId output) {
  impl->outputs.push_back(impl->ids[output]);
  const char *asStr = anchorTypeToString(impl->options.anchorMode);

  // If we are returning EveryN we need to pass in the return period.
  if (impl->options.anchorMode == PopartAnchorTypes::EveryN) {
    impl->anchors.insert(
        {impl->ids[output],
         popart::AnchorReturnType(asStr, impl->options.anchorReturnPeriod)});
  } else {
    impl->anchors.insert({impl->ids[output], popart::AnchorReturnType(asStr)});
  }
}

void Compiler::SetUpInputOp(poptorch::TensorId id, float *ptr,
                            const std::vector<std::int64_t> &dims) {
  // Popart wrapper around the tensor pointer.
  impl->memoryManager.push_back(std::make_unique<popart::NDArrayWrapper<float>>(
      static_cast<float *>(ptr), dims));
  impl->popartIncoming.insert(
      {impl->ids[id], *impl->memoryManager.back().get()});
}

void Compiler::SetUpInputOp(poptorch::TensorId id, std::int32_t *ptr,
                            const std::vector<std::int64_t> &dims) {
  // Popart wrapper around the tensor pointer.
  impl->memoryManager.push_back(
      std::make_unique<popart::NDArrayWrapper<std::int32_t>>(ptr, dims));
  impl->popartIncoming.insert(
      {impl->ids[id], *impl->memoryManager.back().get()});
}

void Compiler::SetUpInputOp(poptorch::TensorId id, std::int64_t *ptr,
                            const std::vector<std::int64_t> &dims) {
  // Popart wrapper around the tensor pointer.
  impl->memoryManager.push_back(
      std::make_unique<popart::NDArrayWrapper<std::int64_t>>(ptr, dims));
  impl->popartIncoming.insert(
      {impl->ids[id], *impl->memoryManager.back().get()});
}

void Compiler::SetUpOutputOp(poptorch::TensorId id, float *ptr,
                             const std::vector<std::int64_t> &dims) {
  // Popart wrapper around the tensor pointer.
  auto memory = std::make_unique<popart::NDArrayWrapper<float>>(
      static_cast<float *>(ptr), dims);

  impl->AddMemoryToOutput(id, ptr, std::move(memory));
}

void Compiler::SetUpOutputOp(poptorch::TensorId id, std::int32_t *ptr,
                             const std::vector<std::int64_t> &dims) {
  // Popart wrapper around the tensor pointer.
  auto memory = std::make_unique<popart::NDArrayWrapper<std::int32_t>>(
      static_cast<std::int32_t *>(ptr), dims);

  impl->AddMemoryToOutput(id, ptr, std::move(memory));
}

void Compiler::SetUpOutputOp(poptorch::TensorId id, bool *ptr,
                             const std::vector<std::int64_t> &dims) {
  // Popart wrapper around the tensor pointer.
  auto memory = std::make_unique<popart::NDArrayWrapper<bool>>(
      static_cast<bool *>(ptr), dims);

  impl->AddMemoryToOutput(id, ptr, std::move(memory));
}

void Compiler::InitSession(const Optimizer &opt) {
  // Try and get a single IPU. If not avaliable, run on CPU.
  // TODO(T22642): Make an actual device selection mechanism.
  std::shared_ptr<popart::DeviceInfo> device =
      popart::DeviceManager::createDeviceManager().acquireAvailableDevice(
          impl->usedIpus.size() * impl->popartOptions.replicatedGraphCount);

  if (!device) {
    logging::warn("No IPU device found, falling back to CPU emulator (IPU "
                  "Model) number of IPUs requested {}",
                  impl->usedIpus.size() *
                      impl->popartOptions.replicatedGraphCount);
    device = popart::DeviceManager::createDeviceManager().createCpuDevice();
  } else {
    logging::debug("Acquired IPU device, running on device.");
  }

  popart::SessionOptions &options = impl->popartOptions;
  bool enableReplicatedGraphs = impl->popartOptions.replicatedGraphCount != 1;
  if (impl->optionsSet.count("enableReplicatedGraphs") &&
      options.enableReplicatedGraphs != enableReplicatedGraphs) {
    logging::warn("enableReplicatedGraphs forced by the user to {}",
                  options.enableReplicatedGraphs);
  } else {
    options.enableReplicatedGraphs = enableReplicatedGraphs;
  }

  logging::info("Popart replication enabled: {} with factor set to {}",
                options.enableReplicatedGraphs, options.replicatedGraphCount);

  // Causes problems with Popart
  const bool constantWeights = false;
  if (impl->optionsSet.count("constantWeights") &&
      options.constantWeights != constantWeights) {
    logging::warn("constantWeights forced by the user to {}",
                  options.constantWeights);
  } else {
    options.constantWeights = constantWeights;
  }

  if (impl->usedIpus.size() > 1) {
    if (!options.enablePipelining) {
      logging::warn("Using {} IPUs but "
                    "poptorch.Options.enablePipelining() is False",
                    impl->usedIpus.size());
    }
    if (impl->optionsSet.count("virtualGraphMode") &&
        options.virtualGraphMode != popart::VirtualGraphMode::Manual) {
      logging::warn("virtualGraphMode forced by the user to {} ",
                    popart::toString(options.virtualGraphMode));
    } else {
      options.virtualGraphMode = popart::VirtualGraphMode::Manual;
    }
  }

  bool enableGradientAccumulation = options.accumulationFactor > 1;
  if (impl->optionsSet.count("enableGradientAccumulation") &&
      !options.enableGradientAccumulation) {
    logging::warn("enableGradientAccumulation forced by the user to {}",
                  options.enableGradientAccumulation);
  } else {
    options.enableGradientAccumulation = enableGradientAccumulation;
  }

  // Create the anchors, these are used to copy to the host.
  auto dataFlow = popart::DataFlow(impl->options.steps, impl->anchors);

  // Create the popart session object to actually run the graph.
  if (!impl->isTraining) {
    // Create an inference session.
    impl->session = popart::InferenceSession::createFromOnnxModel(
        impl->opBuilder->getModelProto(), dataFlow, device, {}, options,
        popart::PatternsLevel::Default);
  } else {
    logging::debug(
        "Adding initial graph optimizer SGD with parameters:: Learning rate "
        "{}, weight decay {}, Momentum {}, Dampening {}",
        opt.learningRate.first, opt.weightDecay.first, opt.momentum.first,
        opt.dampening.first);

    // Create the optimizer from user provided parameters.
    auto optimizer =
        popart::SGD(opt.learningRate, opt.weightDecay, opt.momentum,
                    opt.dampening, {1.0f, true}, // Velocity scaling, off.
                    {1.0f, true});               // Loss scaling, off.

    // Set a global identity loss that all other losses derive from.
    popart::TensorId lossRoot =
        impl->opBuilder->aiGraphcoreOpset1().identityloss(impl->losses);
    impl->opBuilder->virtualGraph(lossRoot, impl->activeIpu);

    // Transform nodes which have training/inference variants. I.E BatchNorm.
    popart::GraphTransformer transformer{impl->opBuilder->getModelProto()};
    transformer.prepareNodesForTraining();

    // Create the training session.
    impl->session = popart::TrainingSession::createFromOnnxModel(
        transformer.getModelProto(), dataFlow, lossRoot, optimizer, device, {},
        options, popart::PatternsLevel::Default);
  }

  logging::trace(
      "Popart serialised IR:\n{}",
      impl->session->serializeIr(popart::IrSerializationFormat::JSON));

  // Poplar compilation.
  try {
    logging::trace("Begining Poplar compilation.");
    impl->session->prepareDevice();
    logging::trace("Finished Poplar compilation.");
  } catch (popart::memory_allocation_err &e) {
    std::ofstream stream;
    stream.open("OOMReport.json");
    stream << e.getGraphReport(true);
    stream.close();

    std::rethrow_exception(std::current_exception());
  }

  if (impl->options.profile) {
    std::ofstream stream;
    stream.open("GraphReport.json");
    stream << impl->session->getGraphReport();
    stream.close();
  }

  // Write the weights immediately after compilation to the IPU.
  CopyWeightsToDevice();
}

// Write the weights into IPU memory from the pytorch tensor buffers in the
// model.
void Compiler::CopyWeightsToDevice() {
  logging::info("Writing weights from host to IPU memory.");
  impl->session->weightsFromHost();
  impl->session->writeWeights(impl->weightCallback);
}

// Read the weights from IPU memory into the pytorch tensor buffers.
void Compiler::CopyWeightsToHost() {
  logging::info("Writing weights from IPU to host.");
  impl->session->weightsToHost();
  impl->session->readWeights(impl->weightCallback);
}

void Compiler::Run(const Optimizer &optimizer) {
  if (optimizer.type != OptimizerType::NONE && impl->isTraining) {
    // Convert the map from the user into a popart SGD class.
    auto newOptimizer = popart::SGD(
        optimizer.learningRate, optimizer.weightDecay, optimizer.momentum,
        optimizer.dampening, {1.0f, true}, // Velocity scaling, off.
        {1.0f, true});                     // Loss scaling, off.

    // Print to debug the new optimizer.
    logging::debug(
        "Updating graph optimizer SGD with parameters: Learning rate "
        "{}, weight decay {}, Momentum {}, Dampening {}",
        optimizer.learningRate.first, optimizer.weightDecay.first,
        optimizer.momentum.first, optimizer.dampening.first);

    // Update the popart graph/poplar executable with the new optimizer.
    popart::TrainingSession &session =
        dynamic_cast<popart::TrainingSession &>(*impl->session.get());
    session.updateOptimizerFromHost(&newOptimizer);
  }

  // Execute the model on IPU.
  popart::StepIO stepio(impl->popartIncoming, impl->popartOutgoing);
  impl->session->run(stepio);

  // In case several outputs point at the same tensor: duplicate the data
  for (auto out : impl->outgoingDuplicates) {
    auto &src = impl->popartOutgoing.at(out.first);
    for (auto ptr : out.second) {
      std::memcpy(ptr, src.data(),
                  src.nelms() *
                      popart::getDataTypeInfoMap().at(src.dataType()).nbytes());
    }
  }
  // The buffers handle the communication between pytorch and popart, we set
  // them up each run.
  impl->popartIncoming.clear();
  impl->popartOutgoing.clear();
  impl->outgoingDuplicates.clear();
  impl->memoryManager.clear();
}

poptorch::PopartTypes Compiler::GetPopartType(poptorch::TensorId tensor) const {
  popart::TensorInfo info = impl->session->getInfo(impl->ids[tensor]);

  if (info.dataType() == popart::DataType::FLOAT) {
    return poptorch::PopartTypes::FLOAT;
  } else if (info.dataType() == popart::DataType::INT32 ||
             info.dataType() == popart::DataType::UINT32) {
    return poptorch::PopartTypes::INT32;
  } else if (info.dataType() == popart::DataType::BOOL) {
    return poptorch::PopartTypes::BOOL;
  }

  ERROR("Unsupported popart type in return: " << info.data_type());
}

bool Compiler::TensorIdIsValid(poptorch::TensorId id) const {
  return id < impl->ids.size();
}

std::vector<std::int64_t> Compiler::GetSize(poptorch::TensorId id) {
  popart::TensorInfo info = impl->session->getInfo(impl->ids[id]);

  return info.shape();
}

void Compiler::SetActiveIpu(std::uint64_t id) { impl->activeIpu = id; }

std::uint64_t Compiler::BatchPerStep() const { return impl->options.steps; }

std::uint64_t Compiler::PopartBatchDim() const {
  return impl->popartOptions.replicatedGraphCount * impl->options.steps *
         impl->popartOptions.accumulationFactor;
}

std::uint64_t Compiler::PopartBatchDimForAnchor(poptorch::TensorId id) const {
  // Get the PopART tensor from our wrapper.
  popart::TensorId popartId = impl->ids[id];

  // Check what the anchor is supposed to return.
  auto iterator = impl->anchors.find(popartId);
  ERROR_ON_MSG(iterator == impl->anchors.end(),
               "Internal Error: Output op doesn't have an anchor.");

  const popart::AnchorReturnType &returnType = iterator->second;

  // If we are returning ALL then we are returning a full batch.
  if (returnType.id() == popart::AnchorReturnTypeId::All) {
    return PopartBatchDim();
  }

  // If we are copying EveryN then we will be returning N.
  if (returnType.id() == popart::AnchorReturnTypeId::EveryN) {
    return PopartBatchDim() / returnType.rp();
  }

  // Return an element for each replica.
  return impl->popartOptions.replicatedGraphCount;
}

Compiler::Compiler(Compiler &&other) { impl = std::move(other.impl); }

Compiler::Compiler(bool isTraining, const SessionOptions &options) {
  impl = std::make_unique<detail::CompilerImpl>();
  impl->isTraining = isTraining;
  impl->popartOptions = options.impl->popartOptions;
  impl->options = options.impl->poptorchOptions;
  impl->optionsSet = options.impl->optionsSet;
}

Compiler::~Compiler() {}

void Compiler::AddOutputType(OutputType type) {
  impl->outputTypes.emplace_back(type);
}

const std::vector<OutputType> &Compiler::OutputTypes() const {
  return impl->outputTypes;
}

SessionOptions::SessionOptions()
    : impl(std::make_unique<detail::SessionOptionsImpl>()) {}

SessionOptions::SessionOptions(SessionOptions &&src)
    : impl(std::move(src.impl)) {}

void SessionOptions::AddStringOption(const char *option, const char *value) {
  impl->Set<std::string>(option, value, impl->stringOptions, "string");
}

void SessionOptions::AddUInt64Option(const char *option, std::uint64_t value) {
  impl->Set(option, value, impl->uint64Options, "uint64");
}

void SessionOptions::AddBoolOption(const char *option, bool value) {
  impl->Set(option, value, impl->boolOptions, "bool");
}

void SessionOptions::AddDoubleOption(const char *option, double value) {
  impl->Set(option, value, impl->doubleOptions, "floating point");
}

void SessionOptions::InsertStringOption(const char *option, const char *value) {
  impl->Set(option, std::pair<std::string, std::string>(value, ""),
            impl->containerOptions, "set / vector");
}

void SessionOptions::InsertStringPairOption(const char *option, const char *key,
                                            const char *value) {
  impl->Set(option, std::pair<std::string, std::string>(key, value),
            impl->containerOptions, "map");
}

SessionOptions::~SessionOptions() {}
} // namespace poptorch
