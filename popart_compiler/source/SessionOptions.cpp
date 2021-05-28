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
#include <popart/popx/devicexmanager.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensors.hpp>

#include "popart_compiler/Compiler.hpp"
#include "popart_compiler/CompilerOptions.hpp"
#include "popart_compiler/PopartEnums.hpp"
#include "popart_compiler/SessionOptions.hpp"
#include "popart_compiler/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace {

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

} // namespace

namespace poptorch {
namespace detail {

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

  registerSetter(uint64_options, "accumulation_and_replication_reduction_type",
                 [&](std::uint64_t value) {
                   ERROR_ON_MSG(value > static_cast<std::uint64_t>(
                                            popart::ReductionType::NoReduction),
                                "Value for popart::ReductionType out of range");
                   popart_options.accumulationAndReplicationReductionType =
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

  registerSetter(string_options, "saveInitializersToFile",
                 [&](const std::string &value) {
                   poptorch_options.external_initializers_file = value;
                 });

  string_options["logDir"] = [&](const std::string &log_dir) {
    UNUSED(log_dir);
    logging::warn(
        "Ignoring call to poptorch.Options._Popart.set(\"logDir\",...): use "
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

  registerSetter(container_options, "lstmOptions",
                 [&](const std::pair<std::string, std::string> &p) {
                   popart_options.lstmOptions.emplace(p);
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
  ADD_POPART_ENUM_OPTION(accumulationAndReplicationReductionType,
                         ReductionType);

  ADD_POPART_STRING_OPTION(logDir);
  ADD_POPART_STRING_OPTION(cachePath);
  ADD_POPART_STRING_OPTION(partialsTypeMatMuls);
  ADD_POPART_STRING_OPTION(customCodeletCompileFlags);
  ADD_POPART_STRING_OPTION(serializedPoprithmsShiftGraphsDir);
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
  ADD_POPART_UINT64_OPTION(defaultPrefetchBufferingDepth);

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
  ADD_POPART_BOOL_OPTION(automaticLossScalingSettings.enabled);
  ADD_POPART_BOOL_OPTION(instrumentWithHardwareCycleCounter);
  ADD_POPART_BOOL_OPTION(enableSupportedDataTypeCasting);
  ADD_POPART_BOOL_OPTION(groupNormStridedChannelGrouping);

  ADD_POPART_DOUBLE_OPTION(outlineSequenceBreakCost);
  ADD_POPART_DOUBLE_OPTION(outlineThreshold);
  ADD_POPART_DOUBLE_OPTION(timeLimitScheduler);
  ADD_POPART_DOUBLE_OPTION(automaticLossScalingSettings.binEdgeLocation);
  ADD_POPART_DOUBLE_OPTION(
      automaticLossScalingSettings.thresholdUpperCountProportion);

#undef ADD_POPART_STRING_OPTION
#undef ADD_POPART_UINT64_OPTION
#undef ADD_POPART_BOOL_OPTION
#undef ADD_POPART_DOUBLE_OPTION
#undef ADD_POPART_ENUM_OPTION
}

} // namespace detail

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
  if (location_tensor == "location_activation") {
    settings = &_impl->popart_options.activationTensorLocationSettings;
  } else if (location_tensor == "location_weight") {
    settings = &_impl->popart_options.weightTensorLocationSettings;
  } else if (location_tensor == "location_optimizer") {
    settings = &_impl->popart_options.optimizerStateTensorLocationSettings;
  } else if (location_tensor == "location_accumulator") {
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

void SessionOptions::setCompilationProgressLogger(
    const std::function<void(int, int)> &logger) {
  _impl->popart_options.compilationProgressLogger = logger;
}

SessionOptions::~SessionOptions() = default;

} // namespace poptorch
