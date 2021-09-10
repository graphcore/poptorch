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
#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

#include "popart_compiler/Compiler.hpp"
#include "popart_compiler/CompilerImpl.hpp"
#include "popart_compiler/MultiConvBuilder.hpp"
#include "popart_compiler/PopartEnums.hpp"
#include "popart_compiler/SessionOptions.hpp"
#include "popart_compiler/Utils.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

#include "CustomOps.hpp"
namespace poptorch {
namespace {

// Helper to let us filter string arguments into const char*s. This is to catch
// the std::string produced by some attributes before they cross the ABI
// boundary.
template <typename T> T convertType(T &&t) { return t; }

std::vector<std::string> convertType(std::vector<const char *> v) {
  return std::vector<std::string>(v.begin(), v.end());
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

  for (auto &param : source_optim.parameters) {
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

  if (source_float && dest_float) {
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
  for (auto &str : strs) {
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
                         const std::vector<std::int64_t> &dims) {
  // Create the tensor info for our new tensor.
  popart::TensorInfo info{type, dims};
  auto popart_id = _impl->active_builder->addInputTensor(info);
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
                               PopartAnchorTypes anchor_mode,
                               size_t anchor_return_period) {
  _impl->outputs.push_back(_impl->ids[output]);

  if (isHostSideConstant(output)) {
    return; // Nothing more to do
  }

  if (anchor_mode == PopartAnchorTypes::N) {
    anchor_mode = _impl->options.anchor_mode;
    if (anchor_mode == PopartAnchorTypes::EveryN) {
      anchor_return_period = _impl->options.anchor_return_period;
    }
  }

  const char *as_str = anchorTypeToString(anchor_mode);

  // If we are returning EveryN we need to pass in the return period.
  if (anchor_mode == PopartAnchorTypes::EveryN) {
    _impl->anchors.insert(
        {_impl->ids[output],
         popart::AnchorReturnType(as_str, anchor_return_period)});
  } else {
    _impl->anchors.insert(
        {_impl->ids[output], popart::AnchorReturnType(as_str)});
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

void Compiler::initSession(const std::vector<Optimizer> &optimizers) {
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

  popart::VirtualGraphMode graph_mode = popart::VirtualGraphMode::Off;
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

    break;
  }
  case detail::ExecutionMode::Sharded: {
    _impl->setOptionIfNotSet(options.enablePipelining, false,
                             "enablePipelining");
    if (_impl->used_ipus.size() > 1) {
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

void Compiler::compileAndExport(const char *filename) {
  ERROR_ON_MSG(!_impl->session,
               "Nothing to export. This may be because the model does not run "
               "any op on the IPU.");

  logging::LogContext ctx_function{"Compiler::compileAndExport"};
  // We use std::ios_base::ate to append to the file: the Python frontend
  // will have already created the folder and written the Poptorch python
  // data in the file.
  //
  // Note: PopArt needs to be able to move backward through the stream so
  // we cannot use std::ios_base::app
  std::fstream stream(filename, std::ios_base::in | std::ios_base::out |
                                    std::ios_base::ate | std::ofstream::binary);
  ERROR_ON_MSG(!stream.is_open(),
               "Failed to open " + std::string(filename) + " for writing");
  stream.seekp(0, std::ios::end);
  logging::LogContext ctx{
      "popart::Session::compileAndExport: Poplar compilation"};
  _impl->session->compileAndExport(stream);
  stream.flush();
  stream.close();
}

void Compiler::loadExecutableAndPrepareDevice(const char *import_filename,
                                              std::int64_t offset) {
  ERROR_ON_MSG(!_impl->session, "Nothing to import. This may be because the "
                                "model does not run any op on an IPU.");

  logging::LogContext ctx{"Compiler::loadExecutableAndPrepareDevice"};
  std::ifstream stream(import_filename, std::ifstream::binary);
  ERROR_ON_MSG(!stream.is_open(), "Failed to open " +
                                      std::string(import_filename) +
                                      " for reading");
  stream.seekg(offset);
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

  // For each individual CPU operation (multiple calls to one op = still one op)
  for (detail::CallbackInternalMetadata &stream_data : _impl->callbacks) {
    const std::uint32_t number_of_inputs = stream_data.input_handles.size();

    // For each input we create a special callback which tracks how many inputs
    // have been added and once they're all in it calls back into python.
    for (std::uint32_t input = 0; input < number_of_inputs; ++input) {
      /*
        Multipurpose callback. Firstly copy from the IPU input buffer into the
        buffer on host. Secondly if this is the last input call, call the
        pytorch host function.
      */
      auto poplar_callback = [=, &stream_data](void *buffer) {
        const std::vector<std::size_t> &shape = stream_data.input_shapes[input];
        std::size_t number_of_elems = 1;
        for (std::size_t elem : shape) {
          number_of_elems = number_of_elems * elem;
        }

        std::size_t size_in_bytes = number_of_elems;

        const poplar::Type type =
            poptorch::poplarTypeFromPoptorch(stream_data.input_types[input]);

        if (type == poplar::UNSIGNED_SHORT || type == poplar::SHORT ||
            type == poplar::HALF) {
          size_in_bytes *= 2;
        } else if (type == poplar::UNSIGNED_INT || type == poplar::INT ||
                   type == poplar::FLOAT) {
          size_in_bytes *= 4;
        } else {
          const bool single_byte_type =
              type == poplar::BOOL || type == poplar::CHAR ||
              type == poplar::SIGNED_CHAR || type == poplar::UNSIGNED_CHAR;

          ERROR_ON_MSG(!single_byte_type, "Unsupported host op type");
        }

        // Copy from IPU into the waiting pytorch tensor on host.
        std::memcpy(reinterpret_cast<char *>(stream_data.input_pointers[input]),
                    reinterpret_cast<char *>(buffer), size_in_bytes);

        // Mark this as another tensor ready.
        stream_data.number_of_input_streams_inited++;

        // Call the callback once all tensors are ready.
        if (stream_data.number_of_input_streams_inited == number_of_inputs) {
          // Call the pytorch function on CPU.
          stream_data.the_callback();

          // Reset counter.
          stream_data.number_of_input_streams_inited = 0;
        }
      };

      // Tell poplar about the callback.
      _impl->session->connectStreamToCallback(stream_data.input_handles[input],
                                              poplar_callback);
    }

    // We then do the outputs, these are much simpler since it is a straight up
    // dependency free data copy.
    for (std::uint32_t output = 0; output < stream_data.output_handles.size();
         ++output) {
      _impl->session->connectStream(stream_data.output_handles[output],
                                    stream_data.output_pointers[output]);
    }
  }

  // Set the random seed (if one was provided) following compilation
  if (_impl->options_set.count("random_seed")) {
    logging::trace("Setting random seed to: {}", _impl->options.random_seed);
    _impl->session->setRandomSeed(_impl->options.random_seed);
  }
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

    // serializeIr must be called after prepareDevice in some cases (e.g.
    // when loading from execution cache)
    logging::trace("Popart serialised IR:\n{}", _impl->getPopartIR());
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

// Write the weights into IPU memory from the pytorch tensor buffers in the
// model.
void Compiler::copyWeightsToDevice(const std::vector<void *> &host_buffers) {
  if (!_impl->session) {
    logging::trace("Skipping writing weights from host to IPU memory.");
    return;
  }

  logging::info("Writing weights from host to IPU memory.");
  _impl->weights.updateData(host_buffers);
  _impl->session->writeWeights(_impl->weights);
  _impl->session->weightsFromHost();
}

// Read the weights from IPU memory into the pytorch tensor buffers.
void Compiler::copyWeightsToHost(const std::vector<void *> &host_buffers) {
  if (!_impl->session) {
    logging::trace("Skipping writing weights from IPU to host.");
    return;
  }

  logging::info("Writing weights from IPU to host.");
  _impl->session->weightsToHost();
  _impl->weights.updateData(host_buffers);
  _impl->session->readWeights(_impl->weights);
}

void Compiler::run(const std::vector<Optimizer> &optimizers) {
  if (!_impl->session) {
    // Nothing to run on IPU
    ERROR_ON(!_impl->popart_incoming.empty());
    ERROR_ON(!_impl->popart_outgoing.empty());
    ERROR_ON(!_impl->outgoing_duplicates.empty());
    ERROR_ON(!_impl->memory_manager.empty());
    return;
  }

  if (!optimizers.empty() && _impl->is_training) {
    std::unique_ptr<popart::Optimizer> optimizer =
        _impl->getPopartOptimizer(optimizers);

    // Update the popart graph/poplar executable with t::attache new optimizer.
    popart::TrainingSession &session =
        dynamic_cast<popart::TrainingSession &>(*_impl->session);
    session.updateOptimizerFromHost(optimizer.get());
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

  // Log the number of cycles if instrumentation is enabled
  popart::SessionOptions &options = _impl->popart_options;
  if (options.instrumentWithHardwareCycleCounter) {
    logging::info("Total number of IPU cycles: {}",
                  _impl->session->getCycleCount());
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

void Compiler::pushNameScope(const char *name) const {
  _impl->active_builder->pushNameScope(std::string(name));
}

void Compiler::popNameScope() const { _impl->active_builder->popNameScope(); }

poptorch::TensorId Compiler::addUntypedInputTensor() {
  popart::TensorId out = _impl->active_builder->addUntypedInputTensor();
  _impl->ids.push_back(out);
  return _impl->ids.size() - 1;
}

void Compiler::assertTensorIs(PopartType dataType,
                              poptorch::TensorId id) const {
  PopartType actual_type = _impl->ids_types.at(id);

  if (__builtin_expect(actual_type == PopartType::UNDEFINED, 0)) {
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
  const std::string op_id =
      "poptorch.host_op_" +
      std::to_string(detail::CallbackInternalMetadata::number_of_added_ops);

  for (std::uint32_t i = 0; i < callback.input_pointers.size(); ++i) {
    const std::string input_id = op_id + ".input." + std::to_string(i);
    metadata.input_handles.push_back(input_id);
  }

  const std::uint32_t num_outputs = callback.output_pointers.size();
  for (std::uint32_t i = 0; i < num_outputs; ++i) {
    const std::string output_id = op_id + ".output." + std::to_string(i);
    metadata.output_handles.push_back(output_id);
  }

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
  attributes_map.insert({"stream_info", to_int64});

  std::vector<popart::TensorId> output = _impl->active_builder->customOp(
      poptorch_custom_ops::host_op, 1, ins, num_outputs, attributes_map);

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

size_t Compiler::getNumInputs() const { return _impl->inputs.size(); }

size_t Compiler::getNumOutputs() const { return _impl->outputs.size(); }

void setPopartLogLevel(logging::Level level) {
  for (uint64_t module = 0;
       module < static_cast<uint64_t>(popart::logging::Module::none);
       module++) {
    popart::logging::setLogLevel(static_cast<popart::logging::Module>(module),
                                 static_cast<popart::logging::Level>(level));
  }
}

namespace detail {
struct ExceptionInfoImpl {
  std::string what;
  std::string type;
  std::vector<std::string> stack;
  std::string filename;
  std::string message;
  uint64_t line;
  std::string recovery_action;
  ErrorCategory category;

  void extractStack(const popart::error *e);
};

void ExceptionInfoImpl::extractStack(const popart::error *e) {
  std::istringstream iss(e->stackreport());
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
    stack.push_back(l.substr(first_space));
  }
}
} // namespace detail

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

ExceptionInfo::ExceptionInfo(const std::exception &e, const char *type,
                             const char *filename, uint64_t line)
    : _impl(std::make_unique<detail::ExceptionInfoImpl>()) {

  _impl->filename = logging::shortPoptorchFilename(filename);
  _impl->line = line;
  _impl->category = ErrorCategory::Other;
  std::string extra_info;
  if (dynamic_cast<const popart::internal_error *>(&e)) {
    _impl->type = "popart_internal_exception";
    _impl->extractStack(dynamic_cast<const popart::error *>(&e));
  } else if (dynamic_cast<const popart::error *>(&e)) {
    _impl->type = "popart_exception";
    _impl->extractStack(dynamic_cast<const popart::error *>(&e));
  } else if (dynamic_cast<const poplar::poplar_error *>(&e)) {
    _impl->type = "poplar_";
    _impl->type += dynamic_cast<const poplar::poplar_error *>(&e)->type;
    if (dynamic_cast<const poplar::link_error *>(&e)) {
      // Note: for some reason this error doesn't set its type in Poplar
      _impl->type = "poplar_link_error";
      extra_info =
          ". Output: " + dynamic_cast<const poplar::link_error *>(&e)->output;
    } else if (dynamic_cast<const poplar::recoverable_runtime_error *>(&e)) {
      _impl->category = ErrorCategory::RuntimeRecoverable;
      _impl->recovery_action = poplar::toString(
          dynamic_cast<const poplar::recoverable_runtime_error *>(&e)
              ->getRecoveryAction());
    } else if (dynamic_cast<const poplar::unrecoverable_runtime_error *>(&e)) {
      _impl->category = ErrorCategory::RuntimeUnrecoverable;
    }
  } else if (dynamic_cast<const poputil::poplibs_error *>(&e)) {
    _impl->type = "poplibs_exception";
  } else {
    if (type != nullptr) {
      _impl->type = type;
    } else {
      _impl->type = "std::exception";
    }
  }
  const std::string &what = e.what();
  if (std::count(what.begin(), what.end(), '\n') > 80) {
    std::ofstream log;
    log.open(ERROR_LOG);
    log << e.what();
    log << extra_info;
    log.close();
    _impl->message = "See " ERROR_LOG " for details";
  } else {
    _impl->message = e.what() + extra_info;
  }
  _impl->what = _impl->message;
}

ErrorCategory ExceptionInfo::category() const { return _impl->category; }

const char *ExceptionInfo::recoveryAction() const {
  return _impl->recovery_action.c_str();
}

const char *ExceptionInfo::filename() const { return _impl->filename.c_str(); }

uint64_t ExceptionInfo::line() const { return _impl->line; }

const char *ExceptionInfo::what() const noexcept { return _impl->what.c_str(); }

const char *ExceptionInfo::type() const { return _impl->type.c_str(); }

int64_t ExceptionInfo::stackDepth() const { return _impl->stack.size(); }

const char *ExceptionInfo::stack(int64_t level) const {
  return _impl->stack.at(level).c_str();
}

ExceptionInfo::~ExceptionInfo() = default;
} // namespace poptorch
