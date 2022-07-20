// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stack>
#include <string>
#include <thread>

#include <popart/graphtransformer.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/optimizer.hpp>
#include <popef/Reader.hpp>
#include <popef/Writer.hpp>
#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

#include "popart_compiler/Compiler.hpp"
#include "popart_compiler/CompilerImpl.hpp"
#include "popart_compiler/MultiConvBuilder.hpp"
#include "popart_compiler/PopartEnums.hpp"
#include "popart_compiler/SessionOptions.hpp"
#include "popart_compiler/Utils.hpp"

#include "poptorch_err/ExceptionInfo.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "CustomOps.hpp"

namespace poptorch {
namespace {

void saveModelProtoIfNeeded(popart::Builder *builder,
                            const char *export_proto_filename) {
  std::string filename = export_proto_filename;
  if (!filename.empty()) {
    // Important: popart_compiler is compiled using C++ 14 and therefore
    // doesn't have access to the filesystem utilities so the caller is
    // responsible for making sure the directories exist and the
    // filename is a valid filename.
    std::ofstream fs(filename);
    bool human_readable = true;
    if (const char *proto_as_bin =
            std::getenv("POPTORCH_EXPORT_PROTO_AS_BINARY")) {
      human_readable = std::stoi(proto_as_bin) == 0;
    }
    if (human_readable) {
      logging::info("Exporting model proto as text (Set "
                    "POPTORCH_EXPORT_PROTO_AS_BINARY=1 to export as binary)");
    } else {
      logging::info("Exporting model proto as binary (Set "
                    "POPTORCH_EXPORT_PROTO_AS_BINARY=0 to export as human "
                    "readable text)");
    }
    fs << builder->getModelProto(human_readable);
    fs.close();
  }
}

// Helper to let us filter string arguments into const char*s. This is to catch
// the std::string produced by some attributes before they cross the ABI
// boundary.
template <typename T> T convertType(T &&t) { return std::forward<T>(t); }

std::vector<std::string> convertType(std::vector<const char *> v) {
  return std::vector<std::string>(v.begin(), v.end());
}

// Convert an overlap string to a PopART TileSet and Exchange Strategy
std::pair<popart::TileSet, popart::ExchangeStrategy>
exchangeStrToPopartEnum(const char *overlap) {
  std::pair<popart::TileSet, popart::ExchangeStrategy> tile_set_and_strat(
      popart::TileSet::Compute, popart::ExchangeStrategy::JustInTime);

  if (strcmp(overlap, "overlap_accumulation_loop") == 0) {
    tile_set_and_strat.first = popart::TileSet::IO;
    tile_set_and_strat.second = popart::ExchangeStrategy::OverlapInnerLoop;
  } else if (strcmp(overlap, "overlap_device_iteration_loop") == 0) {
    tile_set_and_strat.first = popart::TileSet::IO;
    tile_set_and_strat.second = popart::ExchangeStrategy::OverlapLoops;
  } else {
    ERROR_ON(strcmp(overlap, "no_overlap") != 0);
  }

  return tile_set_and_strat;
}

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
    if (!_impl->active_builder->nodeHasAttribute(
            popart::sPipelineStageAttribute, {in}) &&
        !_impl->active_builder->nodeHasAttribute(
            popart::sExecutionPhaseAttribute, {in})) {
      _impl->setExecutionStrategyAttributes({in});
    }

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

// A whitelist of supported loss operations. Popart needs to know which
// operations are losses so they can be marked by the session.
bool IsLoss(const std::string &operation) {
  return operation == "popart::identityloss";
}

} // namespace

void Optimizer::copyParam(const Optimizer &source_optim, const char *source,
                          const char *dest) {
  const float *source_float = nullptr;
  const bool *source_is_const = nullptr;
  float *dest_float = nullptr;
  bool *dest_is_const = nullptr;

  for (const auto &param : source_optim.parameters) {
    const char *param_name = static_cast<const char *>(param.name);
    if (strcmp(param_name, source) == 0) {
      source_float = &param.value;
      source_is_const = &param.is_const;
    }
  }

  for (auto &param : parameters) {
    const char *param_name = static_cast<const char *>(param.name);
    if (strcmp(param_name, dest) == 0) {
      dest_float = &param.value;
      dest_is_const = &param.is_const;
    }
  }

  if ((source_float != nullptr) && (dest_float != nullptr)) {
    ERROR_ON(!source_is_const);
    ERROR_ON(!dest_is_const);

    logging::debug("Set {} ({}) to {} ({})", dest, *dest_float, source,
                   *source_float);

    (*dest_float) = (*source_float);
    (*dest_is_const) = (*source_is_const);
  }
}

PopartAttribute::PopartAttribute(const char *name, const int64_t &value)
    : _name(stringToUniquePtr(name)), _any(new popart::any(value)) {}
PopartAttribute::PopartAttribute(const char *name,
                                 const std::vector<int64_t> &values)
    : _name(stringToUniquePtr(name)), _any(new popart::any(values)) {}
PopartAttribute::PopartAttribute(const char *name, const float &value)
    : _name(stringToUniquePtr(name)), _any(new popart::any(value)) {}
PopartAttribute::PopartAttribute(const char *name,
                                 const std::vector<float> &values)
    : _name(stringToUniquePtr(name)), _any(new popart::any(values)) {}

PopartAttribute::PopartAttribute(const char *name,
                                 const std::unique_ptr<char[]> &str)
    : _name(stringToUniquePtr(name)),
      _any(new popart::any(std::string(str.get()))) {}

PopartAttribute::PopartAttribute(
    const char *name, const std::vector<std::unique_ptr<char[]>> &strs)
    : _name(stringToUniquePtr(name)) {
  std::vector<std::string> strs_new;
  strs_new.reserve(strs.size());
  for (const auto &str : strs) {
    strs_new.emplace_back(str.get());
  }
  _any = std::make_unique<popart::any>(std::move(strs_new));
}

PopartAttribute::PopartAttribute(PopartAttribute &&) = default;
PopartAttribute &PopartAttribute::operator=(PopartAttribute &&) = default;
PopartAttribute::~PopartAttribute() = default;

popart::any *PopartAttribute::getValue() { return _any.get(); }

PopartConstant::PopartConstant(const PopartType &popart_type, const void *data,
                               const std::vector<std::int64_t> &shape) {
  ERROR_ON_MSG(popart_type == PopartType::DOUBLE,
               "Adding a double constant is not supported. "
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
                         const std::vector<std::int64_t> &dims,
                         const char *overlap) {
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{type, dims};
  popart::InputSettings settings;

  auto tile_set_and_strat = exchangeStrToPopartEnum(overlap);
  if (tile_set_and_strat.second != popart::ExchangeStrategy::JustInTime) {
    _impl->using_overlapped_io = true;
  }

  settings.setTileSet(tile_set_and_strat.first);
  settings.setExchangeStrategy(tile_set_and_strat.second);

  auto popart_id = _impl->active_builder->addInputTensor(info, settings);
  _impl->inputs.push_back(popart_id);
  _impl->ids.push_back(popart_id);
  return _impl->ids.size() - 1;
}

poptorch::TensorId Compiler::createTensorId(const char *name) {
  popart::TensorId tensor(name);
  _impl->ids.push_back(tensor);
  return _impl->ids.size() - 1;
}

#define INT_VEC std::vector<std::int64_t>
#define FLOAT_VEC std::vector<float>
#define FLOAT float
#define INT std::int64_t
#define BOOL bool
#define DEBUG_CONTEXT(Name) _impl->getDebugContext(Name)
#define STRING const char *
#define STRING_VEC std::vector<const char *>
#define NONE
#define ARG(Type, Name) , Type Name
#define POPART_CONST_ARG(Name) , const PopartConstant &Name
#define HOST_SIDE_CONST_ARG(Name) , const HostSideConstant &Name
#define POPART_ATTRIB_VEC_ARG(Name)                                            \
  , std::shared_ptr<std::vector<PopartAttribute>> Name
#define BODY_ARG(Name) , convertType(Name)

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
#undef POPART_ATTRIB_VEC_ARG
#undef POPART_CONST_ARG
#undef HOST_SIDE_CONST_ARG
#undef ARG
#undef NONE
#undef STRING_VEC
#undef STRING
#undef BOOL
#undef INT
#undef FLOAT
#undef FLOAT_VEC
#undef INT_VEC
#undef DEBUG_CONTEXT

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

void Compiler::addOutputTensor(poptorch::TensorId output,
                               PopartOutputMode output_mode,
                               size_t output_return_period,
                               const char *overlap) {
  _impl->outputs.push_back(_impl->ids[output]);

  if (isHostSideConstant(output)) {
    return; // Nothing more to do
  }

  if (output_mode == PopartOutputMode::N) {
    output_mode = _impl->options.output_mode;
    if (output_mode == PopartOutputMode::EveryN) {
      output_return_period = _impl->options.output_return_period;
    }
  }

  auto tile_set_and_strat = exchangeStrToPopartEnum(overlap);
  if (tile_set_and_strat.second != popart::ExchangeStrategy::JustInTime) {
    _impl->using_overlapped_io = true;
  }

  // Check for any use of overlapped io
  // NB this relies on the fact that manual anchors never overlap and other
  // outputs all have the same output_mode. If these assumptions change,
  // the logic will have to make sure _impl->using_overlapped_io is correct
  // before any call to this function rather than changed to true on the first
  // instance.
  if (_impl->using_overlapped_io) {
    verifySettingsForOverlappedIO(output_mode);
  }

  const char *as_str = outputModeToString(output_mode);

  // If we are returning EveryN we need to pass in the return period.
  if (output_mode == PopartOutputMode::EveryN) {
    _impl->anchors.insert({_impl->ids[output], popart::AnchorReturnType(
                                                   as_str, output_return_period,
                                                   tile_set_and_strat.first,
                                                   tile_set_and_strat.second)});
  } else {
    _impl->anchors.insert(
        {_impl->ids[output],
         popart::AnchorReturnType(as_str, tile_set_and_strat.first,
                                  tile_set_and_strat.second)});
  }
}

template <typename T>
static void setUpInputImpl(poptorch::TensorId id, T *ptr,
                           const std::vector<std::int64_t> &dims,
                           detail::CompilerImpl *impl) {
  // Popart wrapper around the tensor pointer.
  impl->memory_manager.push_back(
      std::make_unique<popart::NDArrayWrapper<T>>(ptr, dims));
  impl->popart_incoming.insert(
      {impl->ids[id], *impl->memory_manager.back().get()});
}

void Compiler::setUpInputOp(poptorch::TensorId id, float *ptr,
                            const std::vector<std::int64_t> &dims) {
  assertTensorIs(PopartType::FLOAT, id);
  setUpInputImpl(id, ptr, dims, _impl.get());
}

void Compiler::setUpInputOp(poptorch::TensorId id, std::int32_t *ptr,
                            const std::vector<std::int64_t> &dims) {
  assertTensorIs(PopartType::INT32, id);
  setUpInputImpl(id, ptr, dims, _impl.get());
}

void Compiler::setUpInputOp(poptorch::TensorId id, bool *ptr,
                            const std::vector<std::int64_t> &dims) {
  assertTensorIs(PopartType::BOOL, id);
  setUpInputImpl(id, ptr, dims, _impl.get());
}

void Compiler::setUpInputOp(poptorch::TensorId id, std::int8_t *ptr,
                            const std::vector<std::int64_t> &dims) {
  assertTensorIs(PopartType::INT8, id);
  setUpInputImpl(id, ptr, dims, _impl.get());
}

void Compiler::setUpInputOp(poptorch::TensorId id, std::uint8_t *ptr,
                            const std::vector<std::int64_t> &dims) {
  assertTensorIs(PopartType::UINT8, id);
  setUpInputImpl(id, ptr, dims, _impl.get());
}

void Compiler::setUpInputOp(poptorch::TensorId id, std::int16_t *ptr,
                            const std::vector<std::int64_t> &dims,
                            bool float16) {
  if (float16) {
    assertTensorIs(PopartType::FLOAT16, id);
  } else {
    assertTensorIs(PopartType::INT16, id);
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

template <typename T>
static void addOutput(poptorch::TensorId id, T *ptr,
                      const std::vector<std::int64_t> &dims,
                      detail::CompilerImpl *impl) {
  // Popart wrapper around the tensor pointer.
  auto memory =
      std::make_unique<popart::NDArrayWrapper<T>>(static_cast<T *>(ptr), dims);

  impl->addMemoryToOutput(id, ptr, std::move(memory));
}

void Compiler::setUpOutputOp(poptorch::TensorId id, std::uint8_t *ptr,
                             const std::vector<std::int64_t> &dims) {
  addOutput(id, ptr, dims, _impl.get());
}

void Compiler::setUpOutputOp(poptorch::TensorId id, std::int8_t *ptr,
                             const std::vector<std::int64_t> &dims) {
  addOutput(id, ptr, dims, _impl.get());
}

void Compiler::setUpOutputOp(poptorch::TensorId id, float *ptr,
                             const std::vector<std::int64_t> &dims) {
  addOutput(id, ptr, dims, _impl.get());
}

void Compiler::setUpOutputOp(poptorch::TensorId id, std::int32_t *ptr,
                             const std::vector<std::int64_t> &dims) {
  addOutput(id, ptr, dims, _impl.get());
}

void Compiler::setUpOutputOp(poptorch::TensorId id, bool *ptr,
                             const std::vector<std::int64_t> &dims) {
  addOutput(id, ptr, dims, _impl.get());
}

void Compiler::setUpOutputOp(poptorch::TensorId id, std::int16_t *ptr,
                             const std::vector<std::int64_t> &dims) {
  addOutput(id, ptr, dims, _impl.get());
}

void Compiler::initSession(const std::vector<Optimizer> &optimizers,
                           const char *export_proto_filename) {
  logging::LogContext ctx_init_session{"Compiler::initSession"};

  logging::trace("Initializing session");

  // Some simple PyTorch models will not need an IPU at all. However, we do not
  // want users to experience error messages as these may be trivial models
  // which users try in their first use of PopTorch.
  if (_impl->used_ipus.empty()) {
    logging::info("No IPUs are used by this model. This may happen if the "
                  "model is trivial");
    return;
  }

  auto device = _impl->createDevice();
  popart::SessionOptions &options = _impl->popart_options;

  if (options.engineOptions.count("debug.retainDebugInformation") == 0) {
    options.engineOptions.emplace("debug.retainDebugInformation", "false");
    // Message has to be consistent with format used by setOptionIfNotSet()
    logging::debug(
        "engineOptions[debug.retainDebugInformation] set to value false");
  }
  // 'Auto' mode works if only one IPU is used per replica, and allows
  // overlapped IO to work. Excerpt from D51863 in PopART:
  // IO tiles can only be used when virtual graphs are enabled. Virtual graph
  // modes enable to assign tensors and operations to a subset of IPUs, and
  // within each IPU, to a subset of tiles (such as compute and IO tiles). The
  // supported modes are one of: {Manual, Auto, ExecutionPhases}.
  popart::VirtualGraphMode graph_mode = popart::VirtualGraphMode::Auto;
  // If Pipelining wasn't set: enable it if more than 1 IPU is used.
  switch (_impl->options.execution_mode) {
  case detail::ExecutionMode::Pipelined: {
    _impl->setOptionIfNotSet(options.enablePipelining,
                             _impl->used_ipus.size() > 1, "enablePipelining");
    // If we are pipelining we want to turn on recompute by default.
    if (_impl->used_ipus.size() > 1) {
      graph_mode = popart::VirtualGraphMode::Manual;
      _impl->setOptionIfNotSet(
          options.autoRecomputation, popart::RecomputationType::Pipeline,
          "autoRecomputation",
          popart::toString(popart::RecomputationType::Pipeline));
    }

    // TODO(T53152): AccumulateOuterFragmentSchedule::Serial is currently
    // incompatible with gradient clipping and pipelining.
    for (const auto &optimizer : optimizers) {
      if (optimizer.max_grad_norm != std::numeric_limits<float>::infinity()) {
        _impl->setOptionIfNotSet(
            options.accumulateOuterFragmentSettings.schedule,
            popart::AccumulateOuterFragmentSchedule::Scheduler,
            "accumulateOuterFragmentSettings.schedule",
            "AccumulateOuterFragmentSchedule::Scheduler");
        break;
      }
    }

    break;
  }
  case detail::ExecutionMode::Sharded: {
    _impl->setOptionIfNotSet(options.enablePipelining, false,
                             "enablePipelining");
    if (_impl->used_ipus.size() > 1 || _impl->using_overlapped_io) {
      graph_mode = popart::VirtualGraphMode::Manual;
    }
    break;
  }
  case detail::ExecutionMode::Phased: {
    _impl->setOptionIfNotSet(options.enablePipelining, false,
                             "enablePipelining");
    graph_mode = popart::VirtualGraphMode::ExecutionPhases;
    std::uint64_t num_phases = _impl->max_phase + 1;
    std::uint64_t num_stages;
    if (_impl->options.tensors_liveness != detail::Liveness::AlwaysLive) {
      // We want to send the tensors off chip: Tensors stay live through
      // phases N, N+1, N+2 so we need to have a gap of 3 before the bwd
      // pass, otherwise the bwd pass will start in the same phase as the
      // end of the fwd pass.
      num_phases += 3;
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
        popart::TensorStorage::OffChip, "location_activation",
        "useOnChipStorage(False)");
    _impl->setOptionIfNotSet(
        options.weightTensorLocationSettings.location.storage,
        popart::TensorStorage::OffChip, "location_weight",
        "useOnChipStorage(False)");
    _impl->setOptionIfNotSet(
        options.optimizerStateTensorLocationSettings.location.storage,
        popart::TensorStorage::OffChip, "location_optimizer",
        "useOnChipStorage(False)");
    _impl->setOptionIfNotSet(
        options.accumulatorTensorLocationSettings.location.storage,
        popart::TensorStorage::OffChip, "location_accumulator",
        "useOnChipStorage(False)");
    break;
  }
  default:
    ERROR("ExecutionMode not supported");
  }
  // By default allow the user to save / restore the RNG state (It uses slightly
  // more memory).
  _impl->setOptionIfNotSet(options.enableLoadAndOffloadRNGState, true,
                           "enableLoadAndOffloadRNGState");

  _impl->setOptionIfNotSet(options.virtualGraphMode, graph_mode,
                           "virtualGraphMode", popart::toString(graph_mode));

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
    auto num_pipeline_stages = _impl->numPipelineStages();

    if (_impl->is_training) {
      auto num_forward_stages = (num_pipeline_stages + 1) / 2;
      auto num_backward_stages = (num_pipeline_stages - 1) / 2;

      std::stringstream err_msg;
      err_msg << "poptorch.Options().Training.gradientAccumulation must be "
              << "greater than or equal to the number of pipeline stages ("
              << num_pipeline_stages << ") when using "
              << "poptorch.PipelinedExecution. Please note that a model with "
              << num_forward_stages << " pipeline stages in PopTorch will have "
              << "an additional " << num_backward_stages << " stages when "
              << "training.";

      ERROR_ON_MSG(_impl->popart_options.accumulationFactor <
                       static_cast<int64_t>(num_pipeline_stages),
                   err_msg.str());
    } else {
      std::stringstream err_msg;
      err_msg << "poptorch.Options().deviceIterations must be greater than or "
              << "equal to the number of pipeline stages ("
              << num_pipeline_stages << ") when using "
              << "PopTorch.PipelinedExecution.";

      ERROR_ON_MSG(_impl->options.steps < num_pipeline_stages, err_msg.str());
    }
  }

  _impl->setOptionIfNotSet(options.enableGradientAccumulation,
                           options.accumulationFactor > 1,
                           "enableGradientAccumulation");

  // Only explicitly set these options if overlapped I/O are used
  // otherwise we might be overwriting the values set implicitly
  // by some other PopART options (like for example enableExplicitIR()).
  if (_impl->using_overlapped_io) {
    // This is needed for both overlapped IO and explicit pipelining (not yet)
    // supported.
    _impl->setOptionIfNotSet(options.useHostCopyOps, _impl->using_overlapped_io,
                             "useHostCopyOps");

    // This is needed but may cause regressions for existing models. When it is
    // more developed, this will become the default.
    _impl->setOptionIfNotSet(options.enableExplicitMainLoops,
                             _impl->using_overlapped_io,
                             "enableExplicitMainLoops");
  }

  // Create the anchors, these are used to copy to the host.
  auto data_flow = popart::DataFlow(_impl->options.steps, _impl->anchors);

  // Save the initializers to an external file if requested.
  if (!_impl->options.external_initializers_file.empty()) {
    logging::LogContext ctx{"popart::Builder::saveInitializersExternally"};
    logging::trace("Saving initializers to external file {}",
                   _impl->options.external_initializers_file);
    _impl->active_builder->saveInitializersExternally(
        _impl->weights.parameterIds(),
        _impl->options.external_initializers_file);
  }

  auto model_name_set = _impl->options_set.count("model_name") > 0;

  // Tensor location in PopART includes a shardingDomain option which sets
  // which replicas to shard tensors across when using replicated tensor
  // sharding. For now, only one option works for multiple processes, which is
  // to set the type to consecutive across the number of local replica (which
  // is equal to options.replicatedGraphCount on each process).
  //
  // The setting for a single process remains the default (All) which shards
  // tensors across all replica.
  //
  // In future, GCL and PopART will support additional options, which can be
  // exposed to the user.
  if (_impl->options.num_distributed_processes > 1) {
    popart::CommGroup sharding_domain(popart::CommGroupType::Consecutive,
                                      options.replicatedGraphCount);
    options.activationTensorLocationSettings.location.shardingDomain =
        sharding_domain;
    options.weightTensorLocationSettings.location.shardingDomain =
        sharding_domain;
    options.optimizerStateTensorLocationSettings.location.shardingDomain =
        sharding_domain;
    options.accumulatorTensorLocationSettings.location.shardingDomain =
        sharding_domain;
  }

  saveModelProtoIfNeeded(_impl->active_builder, export_proto_filename);

  // Create the popart session object to actually run the graph.
  if (!_impl->is_training) {
    // Create an inference session.
    logging::LogContext ctx{"popart::InferenceSession::createFromOnnxModel"};
    _impl->session = popart::InferenceSession::createFromOnnxModel(
        _impl->active_builder->getModelProto(), data_flow, device, {}, options,
        popart::PatternsLevel::Default,
        model_name_set ? _impl->options.model_name : "inference");
  } else {
    // Create the optimizer from user provided parameters.
    std::unique_ptr<popart::Optimizer> optimizer =
        _impl->getPopartOptimizer(optimizers);

    // Create the training session.
    logging::LogContext ctx{"popart::TrainingSession::createFromOnnxModel"};
    _impl->session = popart::TrainingSession::createFromOnnxModel(
        _impl->active_builder->getModelProto(), data_flow, _impl->loss,
        *optimizer, device, {}, options, _impl->options.patterns,
        model_name_set ? _impl->options.model_name : "training");
  }
}

void Compiler::saveExecutableToFile(const char *export_filename) const {
  ERROR_ON_MSG(!_impl->session,
               "Nothing to export. This may be because the model does not run "
               "any op on the IPU.");

  logging::LogContext ctx_function{"Compiler::saveExecutableToFile"};

  const std::string path(export_filename);
  std::ofstream stream(path, std::ofstream::binary);
  ERROR_ON_MSG(!stream.is_open(), "Failed to open " << path << " for writing");
  logging::LogContext ctx{"popart::Session::saveExecutableToStream"};
  _impl->session->saveExecutableToStream(stream);
  stream.flush();
  stream.close();
}

void Compiler::setRngState(std::uint64_t seed,
                           const std::vector<std::uint32_t> &rng_state) {
  ERROR_ON_MSG(!_impl->session, "Session should be initialised first");
  logging::debug("Setting random seed to: {}", seed);
  if (_impl->session->getIr().getRequiresRandomSeed()) {
    _impl->session->setRandomSeed(seed);
  } else {
    logging::debug("Session has no random behaviour: nothing to do.");
  }
  if (!rng_state.empty()) {
    logging::debug("Setting RNG state");
    _impl->session->setRNGState(rng_state);
  }
}

std::vector<std::uint32_t> Compiler::getRngState() const {
  ERROR_ON_MSG(!_impl->session, "Session should be initialised first");
  logging::debug("Reading RNG state");
  return _impl->session->getRNGState();
}

std::uint64_t Compiler::getRandomSeed() const {
  ERROR_ON_MSG(!_impl->session, "Session should be initialised first");
  logging::debug("Reading random seed");
  if (_impl->session->getIr().getRequiresRandomSeed()) {
    return _impl->session->getRandomSeed();
  }
  logging::debug("Session has no random behaviour: using 0 as seed.");
  return 0;
}

void Compiler::loadExecutableAndPrepareDevice(const char *import_filename) {
  ERROR_ON_MSG(!_impl->session, "Nothing to import. This may be because the "
                                "model does not run any op on an IPU.");

  logging::LogContext ctx{"Compiler::loadExecutableAndPrepareDevice"};

  const std::string path(import_filename);
  auto stream = std::make_shared<std::ifstream>(path, std::ifstream::binary);
  ERROR_ON_MSG(!stream->is_open(), "Failed to open " << path << " for reading");
  _impl->session->loadExecutableFromStream(stream);
  // Don't automatically load the engine: we want to control when this happens
  // to make sure it happens at the same time in distributed environments.
  constexpr bool load_engine = false;
  _impl->session->prepareDevice(load_engine);
  _impl->cachePopartTypes();
}

void Compiler::loadEngineAndConnectStreams() {
  if (!_impl->session) {
    logging::trace("Skipping loading engine");
    return;
  }

  logging::trace("Loading engine");
  _impl->session->loadEngineAndConnectStreams();

  static const std::map<std::reference_wrapper<const poplar::Type>,
                        std::uint8_t, std::less<poplar::Type>>
      host_sizes{// word types
                 {poplar::UNSIGNED_INT, 4},
                 {poplar::INT, 4},
                 {poplar::FLOAT, 4},
                 // half types
                 {poplar::UNSIGNED_SHORT, 2},
                 {poplar::SHORT, 2},
                 {poplar::HALF, 2},
                 // byte types
                 {poplar::BOOL, 1},
                 {poplar::CHAR, 1},
                 {poplar::SIGNED_CHAR, 1},
                 {poplar::UNSIGNED_CHAR, 1}};

  // For each individual CPU operation (multiple calls to one op = still one op)
  for (detail::CallbackInternalMetadata &cb_data : _impl->callbacks) {
    // For each input we create a special callback which tracks how many inputs
    // have been added and once they're all in it calls back into python.
    auto to_size_bytes = [&](const auto &shape, const auto &type) {
      const poplar::Type ptype = poptorch::poplarTypeFromPoptorch(type);

      auto it = host_sizes.find(ptype);
      ERROR_ON_MSG(it == host_sizes.end(), "Unsupported host op type");

      std::size_t number_of_elems = std::accumulate(
          shape.begin(), shape.end(), 1, std::multiplies<std::size_t>());

      return number_of_elems * it->second;
    };

    // Store the amount of data to be transferred for each of the function's
    // input and output arguments.
    std::vector<std::size_t> input_sizes(cb_data.input_shapes.size());
    std::transform(cb_data.input_shapes.begin(), cb_data.input_shapes.end(),
                   cb_data.input_types.begin(), input_sizes.begin(),
                   to_size_bytes);

    std::vector<std::size_t> output_sizes(cb_data.output_shapes.size());
    std::transform(cb_data.output_shapes.begin(), cb_data.output_shapes.end(),
                   cb_data.output_types.begin(), output_sizes.begin(),
                   to_size_bytes);

    auto poplar_callback =
        [input_sizes = std::move(input_sizes),
         output_sizes = std::move(output_sizes),
         &cb_data](const void *const *inputs, size_t number_of_inputs,
                   void *const *outputs, size_t number_of_outputs) {
          ERROR_ON_MSG(number_of_inputs != input_sizes.size(),
                       "Number of inputs does not match");
          ERROR_ON_MSG(number_of_outputs != output_sizes.size(),
                       "Number of outputs does not match");
          ERROR_ON_MSG(inputs == nullptr,
                       "CPU function callback given null inputs");
          ERROR_ON_MSG(outputs == nullptr,
                       "CPU function callback given null outputs");
          ERROR_ON_MSG(number_of_inputs != cb_data.input_pointers.size(),
                       "Number of inputs does not match cb data (got "
                           << cb_data.input_pointers.size() << ")");
          ERROR_ON_MSG(number_of_outputs != cb_data.output_pointers.size(),
                       "Number of outputs does not match cb data (got "
                           << cb_data.output_pointers.size() << ")");
          for (std::size_t input = 0; input < number_of_inputs; ++input) {
            // Copy from IPU into the waiting pytorch tensor on host.
            std::memcpy(reinterpret_cast<char *>(cb_data.input_pointers[input]),
                        reinterpret_cast<const char *>(inputs[input]),
                        input_sizes[input]);
          }
          // Call the pytorch function on CPU.
          cb_data.the_callback();

          // We then do the outputs, these are much simpler since it is a
          // straight up dependency free data copy.
          for (std::size_t output = 0; output < number_of_outputs; ++output) {
            std::memcpy(
                reinterpret_cast<char *>(outputs[output]),
                reinterpret_cast<const char *>(cb_data.output_pointers[output]),
                output_sizes[output]);
          }
        };

    // Tell poplar about the callback.
    _impl->session->connectHostFunction(cb_data.handle,
                                        std::move(poplar_callback));
  }
}

void Compiler::appendPoptorchMetadataToFile(
    const char *serialized_poptorch_metadata, const size_t metadata_length,
    const char *export_filename) {
  popef::Reader reader;
  reader.parseFile(export_filename);
  ERROR_ON_MSG(reader.executables().size() != 1,
               "Popef file does not contain exactly one Executable blob.");
  const std::string &executable_name = reader.executables().at(0).name;

  popef::FileWriter writer(export_filename, popef::FileWriter::Mode::APPEND);
  auto poptorch_blob =
      writer.createOpaqueBlob(poptorch_opaque_name, executable_name);
  poptorch_blob->stream.write(serialized_poptorch_metadata, metadata_length);
  poptorch_blob->close();
  writer.close();
}

std::vector<char>
Compiler::importPoptorchMetadataFromFile(const char *import_filename) {
  popef::Reader reader;
  reader.parseFile(import_filename);

  std::vector<popef::OpaqueReader> opaques = reader.opaqueBlobs();
  auto poptorch_blob_it = std::find_if(
      opaques.begin(), opaques.end(), [](const popef::OpaqueReader &opaque) {
        return opaque.name == poptorch_opaque_name;
      });
  ERROR_ON_MSG(poptorch_blob_it == opaques.end(),
               "Popef file does not contain Poptorch metadata.");

  const size_t buffer_size = poptorch_blob_it->getAvailableReadSize();
  std::vector<char> metadata_buffer(buffer_size);
  poptorch_blob_it->data.read(metadata_buffer.data(), buffer_size);

  return metadata_buffer;
}

void Compiler::compileAndPrepareDevice() {
  if (!_impl->session) {
    logging::trace("Skipping Poplar compilation");

    // This includes host side tensors, so has to be run even without a session.
    _impl->cachePopartTypes();

    return;
  }
  logging::LogContext ctx_func{"Compiler::compileAndPrepareDevice"};

  // Poplar compilation.
  try {
    logging::LogContext ctx{"popart::Session::prepareDevice: Poplar "
                            "compilation"};
    logging::trace("Begining Poplar compilation.");
    constexpr bool load_engine = false;
    // Don't automatically load the engine: we want to control when this happens
    // to make sure it happens at the same time in distributed environments.
    _impl->session->prepareDevice(load_engine);
    logging::trace("Finished Poplar compilation.");
  } catch (popart::memory_allocation_err &e) {
    logging::err("Out of memory, the graph profile is available here: {}",
                 e.getProfilePath());
    std::rethrow_exception(std::current_exception());
  }

  _impl->cachePopartTypes();
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
  const std::string as_string = _impl->getPopartIR();

  // Copy into a memory managed array to get around ABI.
  return stringToUniquePtr(as_string);
}

std::set<std::unique_ptr<char[]>> Compiler::getTensorNames() const {
  std::set<std::unique_ptr<char[]>> casted_ids;

  const auto tensor_ids = _impl->getTensorNames();
  for (const auto &tensor_id : tensor_ids) {
    // Copy into a memory managed array to get around ABI.
    casted_ids.insert(stringToUniquePtr(tensor_id));
  }

  return casted_ids;
}

// Write the weights into IPU memory from the pytorch tensor buffers in the
// model.
void Compiler::copyWeightsToDevice(const std::vector<void *> &host_buffers) {
  if (!_impl->session) {
    logging::trace("Skipping writing weights from host to IPU memory.");
    return;
  }

  logging::info("Writing weights from host to IPU memory.");
  // Do we need to update the host buffers pointers before
  // uploading to the IPU?
  if (!host_buffers.empty()) {
    _impl->weights.updateData(host_buffers);
    _impl->session->writeWeights(_impl->weights);
  }
  _impl->session->weightsFromHost();
}

// Read the weights from IPU memory into the pytorch tensor buffers.
void Compiler::copyWeightsToHost(const std::vector<void *> &host_buffers) {
  if (!_impl->session) {
    logging::trace("Skipping writing weights from IPU to host.");
    return;
  }

  logging::info("Writing weights from IPU to host.");
  // In PopTorch we use copyWeightsToHost and copyWeightsToDevice as
  // synchronisation routines.
  // It means we expect to have one buffer on the host, one on the device and
  // to synchronise the two in one direction or the other.
  //
  // PopART works differently: it has one set of read source buffers and one
  // set of write destination buffers and we need to keep those in sync
  // manually by calling writeWeights()

  // Transfer from the IPU to PopART read source buffers.
  _impl->session->weightsToHost();
  // Update the Poptorch destination buffers
  _impl->weights.updateData(host_buffers);
  // Copy from the PopART read source buffers to the Poptorch buffers.
  _impl->session->readWeights(_impl->weights);
  // Keep the PopART write destination buffer in sync with the PopTorch buffer.
  _impl->session->writeWeights(_impl->weights);
}

void Compiler::updateOptimizers(const std::vector<Optimizer> &optimizers) {
  ERROR_ON(!_impl->session);
  ERROR_ON(optimizers.empty());
  ERROR_ON(!_impl->is_training);

  // Each of the groups of parameters are stored in a single PopART
  // optimizer that's why the vector of optimizers translates into
  // a single PopART optimizer.
  std::unique_ptr<popart::Optimizer> optimizer =
      _impl->getPopartOptimizer(optimizers);

  // Update the popart graph/poplar executable with new optimizer.
  popart::TrainingSession &session =
      dynamic_cast<popart::TrainingSession &>(*_impl->session);
  session.updateOptimizerFromHost(optimizer.get());
}

void Compiler::run() {
  if (!_impl->session) {
    // Nothing to run on IPU
    ERROR_ON(!_impl->popart_incoming.empty());
    ERROR_ON(!_impl->popart_outgoing.empty());
    ERROR_ON(!_impl->outgoing_duplicates.empty());
    ERROR_ON(!_impl->memory_manager.empty());
    return;
  }

  if (!isAttachedToDevice()) {
    attachToDevice();
  }
  // Execute the model on IPU.
  _impl->stepio.populate(_impl->popart_incoming, _impl->popart_outgoing);
  _impl->session->run(_impl->stepio);

  // In case several outputs point at the same tensor: duplicate the data
  for (const auto &out : _impl->outgoing_duplicates) {
    auto &src = _impl->popart_outgoing.at(out.first);
    for (auto *ptr : out.second) {
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

  // Log the number of cycles if instrumentation is enabled
  popart::SessionOptions &options = _impl->popart_options;
  if (options.instrumentWithHardwareCycleCounter) {
    _cycle_count = _impl->session->getCycleCount();
    logging::debug("Total number of IPU cycles: {}", _cycle_count);
  }
}

poptorch::PopartType Compiler::getPopartType(poptorch::TensorId id) const {
  return _impl->getPopartType(id);
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

  if (!_impl->active_builder->hasValueInfo(popart_id)) {
    return {};
  }
  return _impl->active_builder->getTensorShape(popart_id);
}

std::unique_ptr<char[]>
Compiler::getTensorDTypeString(poptorch::TensorId id) const {
  std::string type_str;

  if (_impl->session) {
    type_str = _impl->session->getInfo(_impl->ids[id]).data_type();
  } else {
    auto popart_id = _impl->ids.at(id);
    if (_impl->active_builder->hasValueInfo(popart_id)) {
      type_str = _impl->active_builder->getTensorDtypeString(popart_id);
    } else {
      type_str = "unknown";
    }
  }

  return stringToUniquePtr(type_str);
}

void Compiler::setCurrentPythonCodeLocation(const char *torch_node,
                                            const char *filename,
                                            std::uint64_t line,
                                            std::uint64_t col) {
  UNUSED(col);
  _impl->torch_node = torch_node;
  _impl->code_location = popart::SourceLocation("", filename, line);
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
    } else if (_impl->options.tensors_liveness ==
               detail::Liveness::OffChipAfterFwdNoOverlap) {
      ERROR_ON_MSG(!_impl->options.serial_phases_execution,
                   "This is only supported for serial phase execution");
      _impl->active_phase = phase_id * 2;
    } else {
      _impl->active_phase = phase_id;
    }
    _impl->max_phase = std::max(_impl->active_phase, _impl->max_phase);
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

  // Record a number of times the IPU switches as this is needed to calculate
  // number of pipeline stages.
  if (static_cast<uint64_t>(ipu_id) != _impl->last_ipu_used) {
    _impl->num_ipu_switches++;
  }

  _impl->active_ipu = ipu_id;

  // The previous will revert to -1 but this will remain ipu_id until another
  // IPU is used.
  _impl->last_ipu_used = ipu_id;
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
    const std::vector<std::set<poptorch::TensorId>> &inputs,
    float availableMemoryProportion) {
  for (const auto &ids : inputs) {
    std::set<popart::TensorId> popart_ids;
    std::transform(
        std::begin(ids), std::end(ids),
        std::inserter(popart_ids, std::begin(popart_ids)),
        [this](const poptorch::TensorId &id) { return _impl->ids[id]; });
    _impl->active_builder->setAvailableMemoryProportion(
        popart_ids, availableMemoryProportion);
  }
}

void Compiler::setMatMulSerialization(poptorch::TensorId matmul,
                                      const char *mode, std::uint64_t factor,
                                      std::uint64_t keep_precision) {
  _impl->active_builder->setSerializeMatMul({_impl->ids[matmul]}, mode, factor,
                                            keep_precision != 0u);
}

void Compiler::optimizerGroup(const std::vector<poptorch::TensorId> &inputs,
                              int64_t group) {
  _impl->optimizerGroup(inputs, group);
}

std::vector<TensorMetadata> Compiler::optimizerTensorMetadataList() const {
  std::vector<TensorMetadata> metadata_list;
  auto fn_add_tensor_data = [&](popart::Tensor *t, bool state_tensor) {
    TensorMetadata tm;
    tm.id = t->id.c_str();
    tm.shape = t->info.shape();
    tm.dtype = t->info.data_type().c_str();

    // Optimiser state tensors are variables in PopART, and must be read/written
    // via WeightsIO. Optimiser parameters such as learning rate and loss
    // scaling are either stream or constant tensors, and so can be read/written
    // directly via memcpy
    if (state_tensor) {
      if (!_impl->optim_state_tensors.contains(t->id)) {
        _impl->optim_state_tensors.registerParameter(t->id, t->info);
      }
    } else {
      tm.data = t->tensorData()->data();
      tm.num_bytes = t->info.nbytes();
    }
    metadata_list.push_back(std::move(tm));
  };
  for (auto *t : _impl->session->getIr().optimizerStateTensors()) {
    fn_add_tensor_data(t, true);
  }
  // Note: session->getIr().optimizerTensors() is empty for cached executables,
  // so get the optimizer tensors from the executable instead.
  for (auto *t : _impl->session->getExecutable().getOptimizerTensors()) {
    fn_add_tensor_data(t, false);
  }
  return metadata_list;
}

void Compiler::fillHostOptimizerStateTensorData(
    const std::vector<void *> &host_buffers) {
  logging::info("Writing optimiser state tensors from IPU to host.");
  // In PopTorch we use copyWeightsToHost and copyWeightsToDevice as
  // synchronisation routines.
  // It means we expect to have one buffer on the host, one on the device and
  // to synchronise the two in one direction or the other.
  //
  // PopART works differently: it has one set of read source buffers and one
  // set of write destination buffers and we need to keep those in sync
  // manually by calling writeWeights()

  // Transfer from the IPU to PopART read source buffers.
  _impl->session->weightsToHost();
  // Update the Poptorch destination buffers
  _impl->optim_state_tensors.updateData(host_buffers);
  // Copy from the PopART read source buffers to the Poptorch buffers.
  _impl->session->readWeights(_impl->optim_state_tensors);
  // Keep the PopART write destination buffer in sync with the PopTorch buffer.
  _impl->session->writeWeights(_impl->optim_state_tensors);
}

void Compiler::writeDeviceOptimizerStateTensorData(
    const std::vector<void *> &host_buffers) {
  ERROR_ON_MSG(!_impl->session, "Session should be initialised first");
  ERROR_ON_MSG(!isAttachedToDevice(), "Must be attached to a device to "
                                      "write the optimizer state.");
  logging::info("Writing optimiser state tensors from host to IPU memory.");
  _impl->optim_state_tensors.updateData(host_buffers);
  _impl->session->writeWeights(_impl->optim_state_tensors);
  _impl->session->weightsFromHost();
}

Compiler::Compiler(Compiler &&compiler) : _cycle_count(compiler._cycle_count) {
  _impl = std::move(compiler._impl);
}

Compiler::Compiler(bool is_training, const SessionOptions &options)
    : _cycle_count(no_cycles) {
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

void Compiler::setAttribute(const char *attribute, const char *key,
                            const char *value) {
  _impl->setAttribute(std::string(attribute), std::string(key),
                      std::string(value));
}

void Compiler::clearAttribute(const char *attribute, const char *key) {
  _impl->clearAttribute(std::string(attribute), std::string(key));
}

poptorch::TensorId
Compiler::endForLoop(std::int32_t trip_count, std::int64_t num_outputs,
                     const std::vector<poptorch::TensorId> &inputs) {
  ERROR_ON_MSG(_impl->is_training,
               "poptorch.for_loop() is only supported in inference.");

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

  return HandleOutput<std::vector<popart::TensorId>>{}(output, false,
                                                       _impl.get());
}

void Compiler::pushNameScope(const char *name) {
  _impl->active_builder->pushNameScope(std::string(name));
}

void Compiler::popNameScope() { _impl->active_builder->popNameScope(); }

poptorch::TensorId Compiler::addUntypedInputTensor() {
  popart::TensorId out = _impl->active_builder->addUntypedInputTensor();
  _impl->ids.push_back(out);
  return _impl->ids.size() - 1;
}

void Compiler::assertTensorIs(PopartType dataType,
                              poptorch::TensorId id) const {
  PopartType actual_type = _impl->ids_types.at(id);

  if (__builtin_expect(
          static_cast<std::int64_t>(actual_type == PopartType::UNDEFINED), 0) !=
      0) {
    // Rare case of input tensor never used, so not in IR
    return;
  }

  ERROR_ON_MSG(actual_type != dataType,
               "One or more input data types have changed since the first model"
               " run. You will need to call \"destroy\" on the model before "
               "running with different input data types.");
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

void Compiler::setMultiConvEnableConvDithering(
    const std::vector<int64_t> &conv_ditherings) {
  ERROR_ON_MSG(_impl->multi_conv_builder == nullptr,
               "Unexpected poptorch.MultiConv option: enable_conv_dithering");
  _impl->multi_conv_builder->setEnableConvDithering(conv_ditherings);
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

poptorch::TensorId
Compiler::addCPUCallback(const std::vector<poptorch::TensorId> &inputs,
                         const CallbackMetadata &callback,
                         std::vector<poptorch::PopartType> input_types,
                         std::vector<std::vector<std::size_t>> input_shapes,
                         std::vector<poptorch::PopartType> output_types,
                         std::vector<std::vector<std::size_t>> output_shapes) {
  logging::LogContext ctx{"Compiler::addCPUCallback"};
  logging::trace("Starting CPU callback adding");

  // Usual poptorch -> popart tensor conversion/lookup.
  std::vector<popart::TensorId> ins;
  ins.reserve(inputs.size());
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(ins),
                 [&](poptorch::TensorId index) { return _impl->ids[index]; });

  // Populate the metadata structure which will be used to communicate between
  // all the components involved in running the host op.
  _impl->callbacks.emplace_front();
  detail::CallbackInternalMetadata &metadata = _impl->callbacks.front();

  // Python function we're calling.
  metadata.the_callback = callback.the_callback;

  // Pointers to the waiting python buffers.
  metadata.input_pointers = callback.input_pointers;
  metadata.output_pointers = callback.output_pointers;

  // A tracker so we can see how many streams have been inited by the poplar
  // buffer callback so we can call the python callback once it equals the
  // number of inputs.
  metadata.number_of_input_streams_inited = 0;

  // Used to mangle the name.
  detail::CallbackInternalMetadata::number_of_added_ops++;

  // Create an ID for each op so we can give a unique name to poplar for each
  // output/input.
  metadata.handle =
      "poptorch.host_op_" +
      std::to_string(detail::CallbackInternalMetadata::number_of_added_ops);

  metadata.input_types = std::move(input_types);
  metadata.input_shapes = std::move(input_shapes);
  metadata.output_types = std::move(output_types);
  metadata.output_shapes = std::move(output_shapes);

  std::map<std::string, popart::any> attributes_map;

  // We have to smuggle this through as a pointer as popart attribute map
  // doesn't support generic types.
  detail::CallbackInternalMetadata *as_ptr = &metadata;
  std::intptr_t as_int = reinterpret_cast<std::intptr_t>(as_ptr);

  std::int64_t to_int64 = static_cast<std::int64_t>(as_int);

  logging::trace("Add CPU callback has added pointer {}", to_int64);
  attributes_map.insert({poptorch_custom_ops::host_op_metadata_attr, to_int64});

  std::vector<popart::TensorId> output = _impl->active_builder->customOp(
      poptorch_custom_ops::host_op, 1, ins, metadata.output_types.size(),
      attributes_map);

  // Convert the popart tensors back to poptorch tensors.
  return HandleOutput<decltype(output)>{}(output, false, _impl.get());
}

std::uint32_t detail::CallbackInternalMetadata::number_of_added_ops = 0;

void Compiler::detachFromDevice() { _impl->detachFromDevice(); }

void Compiler::attachToDevice() { _impl->attachToDevice(); }

bool Compiler::isAttachedToDevice() const {
  return _impl->isAttachedToDevice();
}

const std::vector<double> &Compiler::getInputTimestamps(size_t index) const {
  auto id = _impl->inputs[index];
  return _impl->stepio.getInputTimestamps(id);
}

const std::vector<double> &
Compiler::getInputCompleteTimestamps(size_t index) const {
  auto id = _impl->inputs[index];
  return _impl->stepio.getInputCompleteTimestamps(id);
}

const std::vector<double> &Compiler::getOutputTimestamps(size_t index) const {
  auto id = _impl->outputs[index];
  return _impl->stepio.getOutputTimestamps(id);
}

const std::vector<double> &
Compiler::getOutputCompleteTimestamps(size_t index) const {
  auto id = _impl->outputs[index];
  return _impl->stepio.getOutputCompleteTimestamps(id);
}

uint64_t Compiler::getCycleCount() const {
  if (_cycle_count != no_cycles) {
    return _cycle_count;
  }

  ERROR_ON_MSG(!_impl->popart_options.instrumentWithHardwareCycleCounter,
               "Cycle count logging is disabled.");

  ERROR("Please run the model at least once before obtaining cycle count.");
}

size_t Compiler::getNumInputs() const { return _impl->inputs.size(); }

size_t Compiler::getNumOutputs() const { return _impl->outputs.size(); }

void Compiler::verifySettingsForOverlappedIO(PopartOutputMode output_mode) {
  if (_impl->options.execution_mode == detail::ExecutionMode::Pipelined) {
    ERROR("Overlapped IO is not supported with poptorch.PipelinedExecution. "
          "If you are using only one IPU, please switch to "
          "poptorch.ShardedExecution.");
  }

  ERROR_ON_MSG(_impl->popart_options.numIOTiles == 0,
               "No IO tiles allocated. You must allocate at least 32 IO tiles "
               "using poptorch.Options().TensorLocations.numIOTiles.");

  if (output_mode != PopartOutputMode::Sum &&
      output_mode != PopartOutputMode::All) {
    ERROR("Unsupported output mode for overlapped IO. Please switch output "
          "mode to poptorch.OutputMode.All or poptorch.OutputMode.Sum.");
  }
}

void setPopartLogLevel(logging::Level level) {
  for (uint64_t module = 0;
       module < static_cast<uint64_t>(popart::logging::Module::none);
       module++) {
    popart::logging::setLogLevel(static_cast<popart::logging::Module>(module),
                                 static_cast<popart::logging::Level>(level));
  }
}

void throwTestError(TestErrorType type) {
  logging::LogContext ctx_top{"throwTestError::topLevel"};
  {
    logging::LogContext ctx{"throwTestError::bottomLevel"};
    switch (type) {
    case TestErrorType::Poptorch: {
      ERROR("This is a PopTorch error");
    }
    case TestErrorType::Popart: {
      throw popart::error("This is a Popart error");
    }
    case TestErrorType::PopartInternal: {
      throw popart::internal_error("This is a Popart error");
    }
    case TestErrorType::Poplibs: {
      throw poputil::poplibs_error("This is a Poplibs error");
    }
    case TestErrorType::PoplarUnrecoverable: {
      throw poplar::unrecoverable_runtime_error("This is not recoverable");
    }
    case TestErrorType::PoplarUnknown: {
      throw poplar::unknown_runtime_error("Don't know what happened");
    }
    case TestErrorType::PoplarRecoverableFullReset: {
      throw poplar::recoverable_runtime_error(
          poplar::RecoveryAction::FULL_RESET, "Reboot needed");
    }
    case TestErrorType::PoplarLinkError: {
      throw poplar::link_error("Link error",
                               "Library -lfoo not found\ncheck path");
    }
    default: {
      break;
    }
    }
  }
  ERROR("Unknown TestErrorType");
}

namespace {
class PopExceptionInfo : public ExceptionInfo {
public:
  ~PopExceptionInfo() override = default;
  const char *what() const noexcept override;
  const char *type() const override;
  int64_t stackDepth() const override;
  const char *stack(int64_t level) const override;
  const char *filename() const override;
  uint64_t line() const override;
  const char *recoveryAction() const override;
  ErrorCategory category() const override;
  void extractStack(const popart::error &e);

  std::string mwhat;
  std::string mtype;
  std::vector<std::string> mstack;
  std::string mfilename;
  uint64_t mline;
  std::string mrecovery_action;
  ErrorCategory mcategory;
};

const char *PopExceptionInfo::what() const noexcept { return mwhat.c_str(); }

const char *PopExceptionInfo::type() const { return mtype.c_str(); }

int64_t PopExceptionInfo::stackDepth() const { return mstack.size(); }

const char *PopExceptionInfo::stack(int64_t level) const {
  return mstack.at(level).c_str();
}

const char *PopExceptionInfo::filename() const { return mfilename.c_str(); }

uint64_t PopExceptionInfo::line() const { return mline; }

const char *PopExceptionInfo::recoveryAction() const {
  return mrecovery_action.c_str();
}

ErrorCategory PopExceptionInfo::category() const { return mcategory; }

void PopExceptionInfo::extractStack(const popart::error &e) {
  std::istringstream iss(e.stackreport());
  std::string l;
  // PopART adds a numbered prefix to each stack line: remove it:
  // [0] top_level_fn()
  // [1] main()
  //
  // Becomes:
  //
  // top_level_fn()
  // main()
  while (std::getline(iss, l)) {
    size_t first_space = l.find_first_of(' ');
    if (first_space == std::string::npos) {
      first_space = 0;
    } else {
      // Start at the first character after the space
      ++first_space;
    }
    mstack.push_back(l.substr(first_space));
  }
}
} // namespace

void rethrowPopartOrPoplarException(const std::exception_ptr &eptr,
                                    const char *filename, uint64_t line) {
  PopExceptionInfo pei;
  pei.mfilename = logging::shortPoptorchFilename(filename);
  pei.mline = line;
  pei.mcategory = ErrorCategory::Other;
  std::string extra_info;
  try {
    std::rethrow_exception(eptr);
  } catch (const popart::internal_error &ex) {
    pei.mwhat = ex.what();
    pei.mtype = "popart_internal_exception";
    pei.extractStack(ex);
  } catch (const popart::error &ex) {
    pei.mwhat = ex.what();
    pei.mtype = "popart_exception";
    pei.extractStack(ex);
  } catch (const poplar::link_error &ex) {
    // Note: for some reason this error doesn't set its type in Poplar
    pei.mwhat = ex.what();
    pei.mwhat += ". Output: " + ex.output;
    pei.mtype = "poplar_link_error";
  } catch (const poplar::recoverable_runtime_error &ex) {
    pei.mwhat = ex.what();
    pei.mtype = "poplar_";
    pei.mtype += ex.type;
    pei.mcategory = ErrorCategory::RuntimeRecoverable;
    pei.mrecovery_action = poplar::toString(ex.getRecoveryAction());
  } catch (const poplar::unrecoverable_runtime_error &ex) {
    pei.mwhat = ex.what();
    pei.mtype = "poplar_";
    pei.mtype += ex.type;
    pei.mcategory = ErrorCategory::RuntimeUnrecoverable;
  } catch (const poplar::poplar_error &ex) {
    pei.mwhat = ex.what();
    pei.mtype = "poplar_";
    pei.mtype += ex.type;
  } catch (const poputil::poplibs_error &ex) {
    pei.mwhat = ex.what();
    pei.mtype = "poplibs_exception";
  } catch (...) {
    return;
  }
  throw pei;
}

} // namespace poptorch
