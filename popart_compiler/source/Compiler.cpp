// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <fstream>
#include <iostream>
#include <stack>
#include <string>
#include <thread>

#include <popart/graphtransformer.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/optimizer.hpp>

#include "popart_compiler/Compiler.hpp"
#include "popart_compiler/CompilerImpl.hpp"
#include "popart_compiler/MultiConvBuilder.hpp"
#include "popart_compiler/PopartEnums.hpp"
#include "popart_compiler/SessionOptions.hpp"
#include "popart_compiler/Utils.hpp"

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {
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

// A whitelist of supported loss operations. Popart needs to know which
// operations are losses so they can be marked by the session.
bool IsLoss(const std::string &operation) {
  return operation == "popart::identityloss";
}

} // namespace

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
#define POPART_ATTRIB_VEC_ARG(Name)                                            \
  , std::shared_ptr<std::vector<PopartAttribute>> Name
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
#undef POPART_ATTRIB_VEC_ARG
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
}

void Compiler::compileAndExport(const char *filename) {
  logging::LogContext ctx{
      "Compiler::compileAndExport popart::Session::compileAndExport: Poplar "
      "compilation"};
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
  _impl->session->compileAndExport(stream);
  stream.flush();
  stream.close();
}

void Compiler::loadExecutableAndPrepareDevice(const char *import_filename,
                                              std::int64_t offset) {
  logging::LogContext ctx{"Compiler::loadExecutableAndPrepareDevice "};
  std::ifstream stream(import_filename, std::ifstream::binary);
  ERROR_ON_MSG(!stream.is_open(), "Failed to open " +
                                      std::string(import_filename) +
                                      " for reading");
  stream.seekg(offset);
  _impl->session->loadExecutableFromStream(stream);
  _impl->session->prepareDevice();

  // Set the random seed (if one was provided) following compilation
  if (_impl->options_set.count("random_seed")) {
    logging::trace("Setting random seed to: {}", _impl->options.random_seed);
    _impl->session->setRandomSeed(_impl->options.random_seed);
  }
}
void Compiler::compileAndPrepareDevice() {
  // Poplar compilation.
  try {
    logging::LogContext ctx{"Compiler::compileAndPrepareDevice "
                            "popart::Session::prepareDevice: Poplar "
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

  if (!isAttachedToDevice()) {
    attachToDevice();
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

void Compiler::detachFromDevice() { _impl->detachFromDevice(); }

void Compiler::attachToDevice() {
  logging::trace("Begin attaching device");
  _impl->attachToDevice();
  logging::trace("Finished attaching device");
}

bool Compiler::isAttachedToDevice() const {
  return _impl->isAttachedToDevice();
}

} // namespace poptorch
