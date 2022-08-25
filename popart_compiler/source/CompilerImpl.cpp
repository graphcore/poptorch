// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <memory>
#include <numeric>
#include <vector>

#include <popart/adam.hpp>
#include <popart/adaptive.hpp>
#include <popart/builder.hpp>
#include <popart/error.hpp>
#include <popart/half.hpp>
#include <popart/ir.hpp>
#include <popart/op/convbase.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/devicexmanager.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensors.hpp>
#include <poprithms/ndarray/unfold.hpp>
#include <poptorch_logging/Error.hpp>

#include "popart_compiler/Compiler.hpp"
#include "popart_compiler/CompilerImpl.hpp"
#include "popart_compiler/CompilerOptions.hpp"
#include "popart_compiler/MultiConvBuilder.hpp"
#include "popart_compiler/PopartEnums.hpp"
#include "popart_compiler/Utils.hpp"

namespace poptorch {
namespace popart_compiler {
namespace {

std::string toString(const std::vector<std::string> &vec) {
  std::stringstream ss;
  ss << "[";
  std::string sep{};
  for (const auto &s : vec) {
    ss << sep << s;
    sep = ", ";
  }
  ss << "]";
  return ss.str();
}

std::string toString(OptimizerType type) {
  switch (type) {
  case OptimizerType::SGD1:
    return "SGD1";
  case OptimizerType::SGD2:
    return "SGD2";
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
  case OptimizerType::SGD1:
  case OptimizerType::SGD2: {
    if (is_default) {
      return {
          "defaultLearningRate", "defaultWeightDecay",     "defaultMomentum",
          "defaultDampening",    "defaultVelocityScaling", "nesterov",
          "lossScaling"};
    }
    return {"learningRate", "weightDecay",     "momentum",
            "dampening",    "velocityScaling", "nesterov"};
  }
  case OptimizerType::LAMB:
  case OptimizerType::LAMB_NO_BIAS: {
    if (is_default) {
      return {"defaultLearningRate", "defaultWeightDecay",
              "defaultBeta1",        "defaultBeta2",
              "defaultEps",          "defaultMaxWeightNorm",
              "lossScaling"};
    }
    return {"learningRate", "weightDecay", "beta1",
            "beta2",        "eps",         "maxWeightNorm"};
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
  bool use_tf_variant;
  float max_grad_norm;
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
  ss << ", useTfVariant=" << use_tf_variant;
  ss << ", maxGradNorm=" << max_grad_norm;
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
              : popart::DataType::FLOAT),
      use_tf_variant(opt.use_tf_variant), max_grad_norm(opt.max_grad_norm) {
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
  for (const auto &p : opt.parameters) {
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

void assertSingleInstanceMaxNumIPUs(std::size_t num_ipus) {
  ERROR_ON_MSG(num_ipus > 64, "Too many IPUs requested ("
                                  << num_ipus
                                  << "). Experiments that need more than 64 "
                                     "IPUs require distributed execution.");
}

} // namespace

namespace detail {

popart::ConstVoidData StepIO::in(popart::TensorId id, int64_t num_elems,
                                 bool prefetch) {
  (void)prefetch;
  timestamp(&_in_times, id);
  return get<popart::ConstVoidData>(id, &_inputs_info, num_elems);
}

void StepIO::inComplete(popart::TensorId id, int64_t num_elems) {
  (void)num_elems;
  timestamp(&_in_complete_times, id);
}

popart::MutableVoidData StepIO::out(popart::TensorId id, int64_t num_elems) {
  timestamp(&_out_times, id);
  return get<popart::MutableVoidData>(id, &_outputs_info, num_elems);
}

void StepIO::outComplete(popart::TensorId id) {
  timestamp(&_out_complete_times, id);
}

void StepIO::computeStepDataInfo(const popart::TensorId &id,
                                 popart::IArray *array) {
  if (_step_data_info.find(id) != _step_data_info.end()) {
    return;
  }

  auto dtype = AccessorType::getArrayDataType(*array);
  auto rank = AccessorType::getArrayRank(*array);
  std::vector<int64_t> shape;

  for (size_t i = 0; i < rank; ++i) {
    shape.push_back(AccessorType::getArrayDim(*array, i));
  }

  _step_data_info.insert({id, popart::TensorInfo(dtype, shape)});
}

void StepIO::populate(const TensorArrayMap &inputs,
                      const TensorArrayMap &outputs) {
  _inputs_info.clear();
  for (const auto &input : inputs) {
    _inputs_info.insert({input.first, {input.second, 0}});
    _in_times[input.first].clear();
    _in_complete_times[input.first].clear();
    computeStepDataInfo(input.first, &input.second);
  }

  _outputs_info.clear();
  for (const auto &output : outputs) {
    _outputs_info.insert({output.first, {output.second, 0}});
    _out_times[output.first].clear();
    _out_complete_times[output.first].clear();
    computeStepDataInfo(output.first, &output.second);
  }
}

template <typename T>
T StepIO::get(const popart::TensorId &id, TensorArrayInfo *map,
              int64_t num_elems) {
  auto it = map->find(id);
  ERROR_ON_MSG(it == map->end(), "Internal Compiler Error in StepIO");
  auto &array_info = it->second;

  auto it2 = _step_data_info.find(id);
  ERROR_ON_MSG(it2 == _step_data_info.end(),
               "Internal Compiler Error in StepIO");

  T step_data;
  step_data.info = it2->second;

  step_data.data = static_cast<uint8_t *>(AccessorType::getDataPointer(
                       array_info.array)) + // NOLINT
                   array_info.offset;

  int64_t num_bytes =
      static_cast<int64_t>(step_data.info.getDataTypeInfo()->nbytes()) *
      num_elems;

  array_info.offset = (array_info.offset + num_bytes) % step_data.info.nbytes();
  return step_data;
}

void StepIO::timestamp(TensorTimestamps *time, const popart::TensorId &id) {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  auto stamp =
      static_cast<double>(
          std::chrono::duration_cast<std::chrono::milliseconds>(now).count()) /
      1000;
  time->at(id).push_back(stamp);
}

const std::vector<popart::TensorId> &WeightsIO::parameterIds() const {
  return _weights_order;
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
    if (host_buffers[i] != nullptr) {
      _weights[_weights_order[i]].data = host_buffers[i];
    }
  }
}

CompilerImpl::~CompilerImpl() {
  if (_device && isAttachedToDevice()) {
    detachFromDevice();
  }
}

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

std::string CompilerImpl::checkSystemConfig() const {
  ERROR_ON_MSG(num_ipus == 0, "Must call createDevice() first");
  auto dm = popart::DeviceManager::createDeviceManager();
  if (dm.enumerateDevices().empty()) {
    return "\nNo IPU detected in the system: are you sure the gc-driver is "
           "enabled ?";
  }
  if (options_set.count("ipu_id") != 0u) {
    return "";
  }
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
  if (options_set.count("use_model") != 0u) {
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

std::uint64_t CompilerImpl::numPipelineStages() {
  ERROR_ON(options.execution_mode != ExecutionMode::Pipelined);

  // Every time the IPU ID changes, there is an additional stage. In PopTorch,
  // two blocks/stages with the same IPU ID will be merged.
  std::uint64_t forward_stages = num_ipu_switches + 1;

  // If training, there are twice the number of stages for backpropagation
  // minus one (because on the last IPU, the backpropagation happens as part of
  // the same pipeline stage).
  // (NB this is an upper bound, as tensor.detach() could cut off stages, but
  // we ignore unusual edge cases.)
  if (is_training) {
    return forward_stages * 2 - 1;
  }

  return forward_stages;
}

void CompilerImpl::addMemoryToOutput(TensorId id, void *ptr,
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
    const std::vector<popart::TensorId> &tensors) {
  active_builder->addOutputTensor(tensors.at(0));
}

popart::TensorId
CompilerImpl::reshape(const std::vector<popart::TensorId> &tensors,
                      const std::vector<int64_t> &shape) {
  auto ai_onnx = active_builder->aiOnnxOpset10();

  popart::Shape s = {static_cast<int64_t>(shape.size())};
  popart::TensorInfo tensor_info("INT64", s);
  auto new_shape = ai_onnx.constant({shape.data(), tensor_info});
  return ai_onnx.reshape({tensors.at(0), new_shape},
                         getDebugContext("Reshape"));
}

std::vector<popart::TensorId> CompilerImpl::customOperation(
    const std::vector<popart::TensorId> &args, const std::string &op,
    const std::string &domain, std::int64_t version, std::int64_t num_outputs,
    const std::shared_ptr<std::vector<PopartAttribute>> &attributes) {
  logging::info("Adding custom op with {} inputs ",
                static_cast<std::int32_t>(args.size()));

  // Convert to the the format required for Popart. We cannot use popart::any
  // as a known type externally in poptorch to avoid needing popart headers.
  std::map<std::string, popart::any> attributes_map;
  for (auto &attribute : *attributes) {
    attributes_map[attribute.name()] = *(attribute.getValue());
  }

  if (!attributes->empty()) {
    std::stringstream ss;
    ss << "Attributes: ";

    for (auto &attribute : *attributes) {
      ss << attribute.name();

      if (&attribute != &attributes->back()) {
        ss << ", ";
      }
    }
    logging::trace(ss.str().c_str());
  }

  const std::int32_t num_inputs = static_cast<std::int32_t>(args.size());
  popart::OperatorIdentifier id = {domain, op, 1, num_inputs};

  return active_builder->customOp(id, version, args, num_outputs,
                                  attributes_map, getDebugContext(op));
}

popart::TensorId CompilerImpl::recomputationCheckpoint(
    const std::vector<popart::TensorId> &tensors) {
  // Popart is simply a for loop over vector inputs and it is better for the
  // PyTorch Graph to avoid Tuple/List packs and unpacks
  ERROR_ON(tensors.size() != 1);
  return active_builder->checkpointOutput(tensors)[0];
}

popart::TensorId
CompilerImpl::tensorConstant(const std::vector<popart::TensorId> &tensors,
                             const PopartConstant &constant) {
  UNUSED(tensors);
  auto ai_onnx = active_builder->aiOnnxOpset10();

  return ai_onnx.constant(constant.getPopartData(),
                          getDebugContext("Constant"));
}

TensorId CompilerImpl::hostSideTensorConstant(
    const std::vector<popart::TensorId> &tensors, HostSideConstant constant) {
  UNUSED(tensors);
  _host_side_constants.emplace(std::make_pair(ids.size(), std::move(constant)));

  // Add a dummy into ids
  ids.emplace_back("__poptorch__host_side_constant");

  return ids.size() - 1;
}

std::shared_ptr<popart::DeviceInfo>
CompilerImpl::createDevice(bool must_attach) {
  auto connection_type = options.connection_type;
  if (must_attach) {
    ERROR_ON_MSG(
        connection_type == popart::DeviceConnectionType::Never,
        "[Internal] must_attach incompatible with connection type Never");
    connection_type = popart::DeviceConnectionType::Always;
    _device = nullptr;
  } else {
    ERROR_ON_MSG(_device, "device already created");
  }
  updateUseModelConfig();
  ERROR_ON(used_ipus.empty());

  // Sometimes phased execution doesn't use all of the IPUs in a range, so check
  // the Ids too.
  auto max_ipu_id = *std::max_element(used_ipus.begin(), used_ipus.end());
  num_ipus = std::max(used_ipus.size(), max_ipu_id + 1) *
             popart_options.replicatedGraphCount;
  ERROR_ON_MSG(num_ipus == 0, "Your compiled model is empty (All the "
                              "operations have been optimised out)");
  assertSingleInstanceMaxNumIPUs(num_ipus);
  if (options.ipu_model) {
    if (popart_options.enableEngineCaching) {
      logging::warn("enableExecutableCaching doesn't work with the IPU model");
    }

    errorOnCycleLogging();

    std::map<std::string, std::string> model_options;
    model_options["numIPUs"] = std::to_string(num_ipus);
    std::string env_ipu_model_version = getIpuModelVersion();
    model_options["ipuVersion"] = env_ipu_model_version;
    int num_tiles_per_ipu = getNumTilesPerIpu(env_ipu_model_version);
    model_options["tilesPerIPU"] = std::to_string(num_tiles_per_ipu);

    ERROR_ON_MSG(connection_type == popart::DeviceConnectionType::Never,
                 "ConnectionType.Never / poptorch.Options.useOfflineIpuTarget "
                 "not supported for the IPU model");
    _device = popart::DeviceManager::createDeviceManager().createIpuModelDevice(
        model_options);
    // Acquired HW devices will be attached if the used connection type is
    // Always but createIpuModelDevice() doesn't take a connection type
    // so we manually attach to the device if the connection type is needed.
    if (connection_type == popart::DeviceConnectionType::Always) {
      ERROR_ON_MSG(!_device->attach(), "Internal error: attach can't fail for "
                                       "model devices");
    }
    logging::debug("Instantiated device, running on IPU model with {} tiles.",
                   num_tiles_per_ipu);
  } else {
    if (connection_type == popart::DeviceConnectionType::Never) {
      // Offline compilation path: create an offline device regardless of what's
      // present on the system.
      ERROR_ON_MSG(options_set.count("ipu_id"),
                   "Offline compilation targeting a specific id not supported");
      errorOnCycleLogging();

      const auto get_ipu_version = [&]() -> std::int64_t {
        if (options.ipu_version == CompilerOptions::use_system_ipu_version) {
          return ipuHardwareVersion();
        }
        return options.ipu_version;
      };

      std::map<std::string, std::string> device_options;
      device_options["numIPUs"] = std::to_string(num_ipus);
      device_options["ipuVersion"] = "ipu" + std::to_string(get_ipu_version());
      device_options["syncPattern"] =
          popart::syncPatternToString(options.sync_pattern);
      _device =
          popart::DeviceManager::createDeviceManager().createOfflineIPUDevice(
              device_options);
      ERROR_ON_MSG(!_device, "Failed to create offline IPU device");
    } else {
      // Round up number of ipus to a power of 2.
      auto rounded_num_ipus = roundUpNumIPUs(num_ipus);

      if (rounded_num_ipus != num_ipus) {
        std::string common_msg(", because PopTorch must reserve a power of 2 or"
                               " maximum of 64 IPUs per process");
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
                << ". Please reconfigure your model to use a "
                   "different number of IPUs or set "
                   "poptorch.Options().autoRoundNumIPUs(True).");
        }
      }
      assertSingleInstanceMaxNumIPUs(num_ipus);
      do {
        // Regular IPU hardware target
        if (options_set.count("ipu_id") == 0u) {
          _device = popart::DeviceManager::createDeviceManager()
                        .tryAcquireAvailableDevice(
                            num_ipus, 0, options.sync_pattern, connection_type);
          ERROR_ON_MSG(!_device && !waitIfUnavailable(),
                       "Failed to acquire " << num_ipus << " IPU(s)"
                                            << this->checkSystemConfig());
          if (_device) {
            logging::debug("Acquired {} IPU(s): running on device Id {}.",
                           num_ipus, _device->getId());
          }
        } else {
          _device =
              popart::DeviceManager::createDeviceManager().tryAcquireDeviceById(
                  options.ipu_id, options.sync_pattern, connection_type);
          ERROR_ON_MSG(!_device && !waitIfUnavailable(),
                       "Failed to acquire device Id " << options.ipu_id
                                                      << checkSystemConfig());
          ERROR_ON_MSG(_device && static_cast<std::uint64_t>(
                                      _device->getNumIpus()) < num_ipus,
                       "Expected at least replication factor * used IPUs = "
                           << used_ipus.size() << " * "
                           << popart_options.replicatedGraphCount << " = "
                           << num_ipus << " device Ids but the user provided "
                           << _device->getNumIpus());
          if (_device &&
              static_cast<std::uint64_t>(_device->getNumIpus()) != num_ipus) {
            logging::warn(
                "Expected replication factor * used IPUs = {} * {} "
                "= {} device Ids but the device selected has {} IPUs which "
                "means some of them will not be used.",
                used_ipus.size(), popart_options.replicatedGraphCount, num_ipus,
                _device->getNumIpus());
          }
          if (_device) {
            logging::debug("Acquired IPU device with id {}, running on device.",
                           options.ipu_id);
          }
        }
      } while (!_device && waitForAWhile());
    }
  }
  if (_device->isAttached()) {
    logging::trace("Attached to device {}", _device->getId());
  }
  return _device;
}

void CompilerImpl::detachFromDevice() {
  if (used_ipus.empty()) {
    return;
  }

  logging::trace("Begin detaching device {}", _device->getId());
  ERROR_ON_MSG(!_device, "Cannot find a valid device");
  ERROR_ON_MSG(!_device->isAttached(), "The device has already been detached");
  _device->detach();
  logging::debug("Detached from device {}", _device->getId());
}

bool CompilerImpl::isAttachedToDevice() const {
  if (used_ipus.empty()) {
    // We are always attached to at least 0 IPUs.
    return true;
  }

  ERROR_ON_MSG(!_device, "Cannot find a valid device");
  return _device->isAttached();
}

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
CompilerImpl::getPopartOptimizer(std::vector<Optimizer> optimizers) {
  if (optimizers.empty()) {
    return nullptr;
  }

  // If using the separate tensor variant, glue velocity scaling to loss
  // scaling. When T39344 is completed, there will be no benefit to setting
  // velocity scaling different to loss scaling for the separate tensor case.

  // The first optimizer contains the default values.
  auto &default_value_optimizer(optimizers[0]);

  if (default_value_optimizer.type == OptimizerType::SGD2) {
    copyParam(default_value_optimizer, default_value_optimizer, "lossScaling",
              "velocityScaling");
  }

  // The first optimizer contains the default values.
  OptimizerParameters opt{optimizers[0], true};

  // Print to debug the new optimizer.
  logging::debug("Updating graph optimizer with {}", opt.debug());

  ERROR_ON_MSG(std::isnan(opt.max_grad_norm),
               "Maximum norm of gradients cannot be NaN");

  std::vector<popart::ClipNormSettings> clipnorms;
  if (opt.max_grad_norm != std::numeric_limits<float>::infinity()) {
    clipnorms.push_back(
        popart::ClipNormSettings::clipAllWeights(opt.max_grad_norm));
  }

  switch (opt.type) {
  case OptimizerType::SGD1: {
    ERROR_ON(!opt.accum_types_provided);
    auto optimizer = std::unique_ptr<popart::SGD>(new popart::SGD(
        opt.params, clipnorms, popart::SGDAccumulatorAndMomentum::Combined,
        popart::DataType::UNDEFINED, popart::DataType::UNDEFINED,
        getDebugContext("SGD")));
    updateGroups(optimizer.get(), optimizers);
    return optimizer;
  }
  case OptimizerType::SGD2: {
    ERROR_ON(!opt.accum_types_provided);

    // Copy loss scaling to velocity scaling for all groups
    for (std::size_t idx = 1; idx < optimizers.size(); ++idx) {
      copyParam(optimizers[idx], default_value_optimizer, "lossScaling",
                "velocityScaling");
    }

    auto optimizer = std::unique_ptr<popart::SGD>(new popart::SGD(
        opt.params, clipnorms, popart::SGDAccumulatorAndMomentum::Separate,
        opt.accum_type, opt.first_order_momentum_accum_type,
        getDebugContext("SGD")));
    updateGroups(optimizer.get(), optimizers);
    return optimizer;
  }
  case OptimizerType::ADAM:
  case OptimizerType::ADAMW:
  case OptimizerType::ADAMW_NO_BIAS:
  case OptimizerType::LAMB:
  case OptimizerType::LAMB_NO_BIAS: {
    auto adam_mode = popart::AdamMode::Adam;
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
        opt.second_order_momentum_accum_type, clipnorms, false,
        getDebugContext("Adam"));
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
        opt.second_order_momentum_accum_type, popart::DataType::FLOAT,
        opt.use_tf_variant, getDebugContext("Adaptive"));
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
  popart::TensorId output = ai_onnx.add(in, getDebugContext("AddNotInPlace"));
  active_builder->setInplacePreferences(
      output, {{"AddLhsInplace", -1}, {"AddRhsInplace", -1}});
  return output;
}

popart::TensorId
CompilerImpl::randomNormal(const std::vector<popart::TensorId> &tensors,
                           const std::vector<int64_t> &shape, float mean,
                           float scale, const std::string &dtype) {
  UNUSED(tensors);
  auto ai_onnx = active_builder->aiOnnxOpset10();
  auto pdt = popart::dataTypeFromString(dtype);
  return ai_onnx.randomnormal(shape, popart::getONNXDataTypeAsInt(pdt), mean,
                              scale, nonstd::optional<float>(),
                              getDebugContext("Randomnormal"));
}

popart::TensorId
CompilerImpl::randomUniform(const std::vector<popart::TensorId> &tensors,
                            const std::vector<int64_t> &shape, float high,
                            float low, const std::string &dtype) {
  UNUSED(tensors);
  auto ai_onnx = active_builder->aiOnnxOpset10();
  auto pdt = popart::dataTypeFromString(dtype);
  return ai_onnx.randomuniform(shape, popart::getONNXDataTypeAsInt(pdt), high,
                               low, nonstd::optional<float>(),
                               getDebugContext("Randomuniform"));
}

popart::TensorId
CompilerImpl::ones(const std::vector<popart::TensorId> &tensors,
                   const std::vector<int64_t> &shape,
                   const std::string &dtype) {
  return zerosOrOnes(tensors, shape, dtype, false);
}

popart::TensorId
CompilerImpl::zeros(const std::vector<popart::TensorId> &tensors,
                    const std::vector<int64_t> &shape,
                    const std::string &dtype) {
  return zerosOrOnes(tensors, shape, dtype, true);
}

popart::TensorId
CompilerImpl::zerosOrOnes(const std::vector<popart::TensorId> &tensors,
                          const std::vector<int64_t> &shape,
                          const std::string &dtype, bool zeros) {
  auto total_size = static_cast<size_t>(std::accumulate(
      shape.begin(), shape.end(), 1, std::multiplies<size_t>()));

  if (dtype == "INT32") {
    std::vector<int32_t> const_buff(total_size, zeros ? 0 : 1);
    PopartConstant popart_const(PopartType::INT32, &const_buff[0], shape);
    return tensorConstant(tensors, popart_const);
  }
  if (dtype == "FLOAT") {
    std::vector<float> const_buff(total_size, zeros ? 0 : 1);
    PopartConstant popart_const(PopartType::FLOAT, &const_buff[0], shape);
    return tensorConstant(tensors, popart_const);
  }
  if (dtype == "FLOAT16") {
    std::vector<uint16_t> const_buff(total_size,
                                     popart::floatToHalf(zeros ? 0 : 1));
    PopartConstant popart_const(PopartType::FLOAT16, &const_buff[0], shape);
    return tensorConstant(tensors, popart_const);
  }
  if (dtype == "BOOL") {
    struct Bool {
      bool b;
    };
    std::vector<Bool> const_buff(total_size, {!zeros});
    PopartConstant popart_const(PopartType::BOOL, &(const_buff[0].b), shape);
    return tensorConstant(tensors, popart_const);
  }
  ERROR("Unsupported type " << dtype);
}

popart::TensorId
CompilerImpl::unfold(const std::vector<popart::TensorId> &tensors,
                     int64_t dimension, int64_t size, int64_t step) {
  // Implements the TUnfoldHelper interface in Poprithms using ONNX operations.
  struct PoptorchUnfoldHelper {
    struct InternalState {
      CompilerImpl *parent;
      popart::Builder *builder;
      popart::TensorId tensor;

      popart::TensorId scalarConstI64(int64_t val) const {
        PopartConstant val_const(PopartType::INT64, &val, {});
        return parent->tensorConstant({}, val_const);
      }

      popart::TensorId shapeAsTensor(const std::vector<uint64_t> &shape) const {
        std::vector<int64_t> new_shape(shape.begin(), shape.end());
        PopartConstant shape_const(PopartType::INT64, new_shape.data(),
                                   {static_cast<int64_t>(new_shape.size())});
        return parent->tensorConstant({}, shape_const);
      }

      InternalState transform(popart::TensorId &&new_id) const {
        InternalState new_state(*this);
        new_state.tensor = std::move(new_id);
        parent->setExecutionStrategyAttributes({new_state.tensor});
        return new_state;
      }
    };

    static InternalState slice(const InternalState &state, uint64_t dim,
                               uint64_t start, uint64_t end) {
      auto dims = state.scalarConstI64(dim);
      auto starts = state.scalarConstI64(start);
      auto ends = state.scalarConstI64(end);

      return state.transform(state.builder->aiOnnxOpset10().slice(
          {state.tensor, starts, ends, dims}));
    }

    static InternalState broadcast(const InternalState &state, uint64_t N,
                                   uint64_t dim) {
      auto new_shape = shape(state);
      ERROR_ON(new_shape[dim] != 1);
      new_shape[dim] *= N;
      auto shape_tensor = state.shapeAsTensor(new_shape);

      return state.transform(
          state.builder->aiOnnxOpset10().expand({state.tensor, shape_tensor}));
    }

    static InternalState reshape(const InternalState &state,
                                 const std::vector<uint64_t> &shape) {
      auto shape_tensor = state.shapeAsTensor(shape);

      return state.transform(
          state.builder->aiOnnxOpset10().reshape({state.tensor, shape_tensor}));
    }

    static InternalState concat(const std::vector<InternalState> &states,
                                uint64_t axis) {
      ERROR_ON(states.empty());

      std::vector<popart::TensorId> tensor_ids;
      tensor_ids.reserve(states.size());
      for (const auto &tensor : states) {
        tensor_ids.push_back(tensor.tensor);
      }

      const auto &first = states.front();
      return first.transform(first.builder->aiOnnxOpset10().concat(
          tensor_ids, static_cast<int64_t>(axis)));
    }

    static InternalState dimShuffle(const InternalState &state,
                                    const std::vector<uint64_t> &permutation) {
      std::vector<int64_t> permutation_ints(permutation.begin(),
                                            permutation.end());
      state.builder->setAttribute("perm", permutation_ints);
      auto new_tensor = state.transform(
          state.builder->aiOnnxOpset10().transpose({state.tensor}));
      state.builder->clearAttribute("perm");
      return new_tensor;
    }

    static uint64_t dim(const InternalState &state, uint64_t axis) {
      return state.builder->getTensorShape(state.tensor)[axis];
    }

    static uint64_t rank_u64(const InternalState &state) { // NOLINT
      return static_cast<uint64_t>(
          state.builder->getTensorShape(state.tensor).size());
    }

    static std::vector<uint64_t> shape(const InternalState &state) {
      const auto &&my_shape = state.builder->getTensorShape(state.tensor);
      return std::vector<uint64_t>(my_shape.begin(), my_shape.end());
    }
  };

  ERROR_ON(dimension < 0);
  ERROR_ON(size < 0);
  ERROR_ON(step < 0);
  ERROR_ON(tensors.size() != 1);
  const auto &first = tensors.front();

  using T = PoptorchUnfoldHelper::InternalState;
  using H = PoptorchUnfoldHelper;

  return poprithms::ndarray::Unfolder<T, H>::unfold(
             {this, active_builder, first}, dimension, size, step)
      .tensor;
}

popart::TensorId CompilerImpl::prelu(std::vector<popart::TensorId> &tensors) {
  const popart::TensorId &self = tensors[0];
  popart::TensorId &weight = tensors[1];

  const auto self_shape = active_builder->getTensorShape(self);
  const auto weight_shape = active_builder->getTensorShape(weight);

  if (self_shape.size() > weight_shape.size() + 1) {
    // PyTorch's implementation adds some extra singleton dimensions to the
    // weight to ensure it is 'unidirectionally broadcastable' with the input.
    std::vector<std::int64_t> unsqueeze_axes(self_shape.size() -
                                             weight_shape.size() - 1);
    std::iota(unsqueeze_axes.begin(), unsqueeze_axes.end(),
              weight_shape.size());
    weight = active_builder->aiOnnxOpset10().unsqueeze(
        {weight}, unsqueeze_axes, getDebugContext("Unsqueeze"));
    setExecutionStrategyAttributes({weight});
  }

  return active_builder->aiOnnxOpset10().prelu(tensors,
                                               getDebugContext("Prelu"));
}

const HostSideConstant &CompilerImpl::getHostSideConstant(TensorId id) const {
  return _host_side_constants.at(id);
}

bool CompilerImpl::isHostSideConstant(TensorId id) const {
  return _host_side_constants.count(id) != 0u;
}

void CompilerImpl::addMultiConvPart(
    const std::vector<popart::TensorId> &tensors,
    const std::vector<int64_t> &dilations,
    const std::vector<int64_t> &kernel_shape, const std::vector<int64_t> &pads,
    const std::vector<int64_t> &strides) {
  if (multi_conv_builder == nullptr) {
    multi_conv_builder = std::make_unique<MultiConvBuilder>();
  }

  multi_conv_builder->addConv(tensors, dilations, kernel_shape, pads, strides);
}

void CompilerImpl::setAttribute(const std::string &attribute,
                                const std::string &key,
                                const std::string &value) {
  auto &attrs = _attribute_key_value_map[attribute];
  attrs[key] = value;
  std::vector<std::string> attrs_vec;
  for (auto &attr : attrs) {
    attrs_vec.push_back(attr.first);
    attrs_vec.push_back(attr.second);
  }
  active_builder->setAttribute(attribute, attrs_vec);
}

void CompilerImpl::clearAttribute(const std::string &attribute,
                                  const std::string &key) {
  auto &attrs = _attribute_key_value_map[attribute];
  ERROR_ON_MSG(attrs.erase(key) == 0, "Unknown key '" << key
                                                      << "' for attribute '"
                                                      << attribute << "'.");
  active_builder->clearAttribute(attribute);
  if (attrs.empty()) {
    ERROR_ON_MSG(_attribute_key_value_map.erase(attribute) == 0,
                 "Unknown attribute '" << attribute << "'.");
  } else {
    std::vector<std::string> attrs_vec;
    for (auto &attr : attrs) {
      attrs_vec.push_back(attr.first);
      attrs_vec.push_back(attr.second);
    }
    active_builder->setAttribute(attribute, attrs_vec);
  }
}

popart::DebugContext CompilerImpl::getDebugContext(const std::string &name) {
  std::string op_name = op_builder->getNameScope() + name;
  popart::DebugContext dc(name, code_location);
  popart::DebugInfo di(dc, "poptorch");
  di.setValue("torch_schema", torch_node);
  di.setValue("op_type", name);
  di.setValue("op_name", op_name);
  return {di};
}

std::vector<popart::TensorId> CompilerImpl::endMultiConv() {
  ERROR_ON_MSG(multi_conv_builder == nullptr, "Unexpected end_multi_conv.");
  auto outs = multi_conv_builder->build(active_builder);
  multi_conv_builder.reset();
  return outs;
}

bool CompilerImpl::waitIfUnavailable() const {
  // Force disable the wait if the system doesn't contain an IPU that
  // matches the requested config.
  static const bool should_wait =
      waitIfIpuIsUnavailable() && checkSystemConfig().empty();
  return should_wait;
}

void CompilerImpl::attachToDevice() {
  if (used_ipus.empty()) {
    // We are always attached to at least 0 IPUs.
    return;
  }

  ERROR_ON_MSG(_device->isAttached(), "Already attached to a device");

  // TODO(T21799): PopART onDemand connection will only try to connect to
  // the first device matching the requested config which means if several
  // tests only need 1 IPU, they will all wait on IPU 0.
  // As a workaround we request a new device from PopART and swap the device
  // in the live session.
  session->getDevice().setDeviceInfo(createDevice(/*must_attach*/ true));

  ERROR_ON_MSG(!_device, "Cannot find a valid device");
  ERROR_ON_MSG(!_device->isAttached(), "Still not attached to a device");
  session->getDevice().loadEngineAndConnectStreams();
}

std::string CompilerImpl::getPopartIR() const {
  if (used_ipus.empty()) {
    return "unavailable (No IPUs used)";
  }

  if (session->getExecutable().isDeserialized()) {
    return "unavailable (Cached executable)";
  }
  return session->serializeIr(popart::IrSerializationFormat::JSON);
}

std::set<popart::TensorId> CompilerImpl::getTensorNames() const {
  return session->getAllTensorIds();
}

PopartType CompilerImpl::getPopartType(TensorId id) const {
  if (isHostSideConstant(id)) {
    return getHostSideConstant(id).popartType();
  }

  popart::DataType dtype;
  auto popart_id = ids[id];
  if (!session) {
    if (!active_builder->hasValueInfo(popart_id)) {
      return PopartType::UNDEFINED;
    }
    dtype = active_builder->getTensorDataType(popart_id);
  } else {
    if (!session->hasInfo(popart_id)) {
      return PopartType::UNDEFINED;
    }
    popart::TensorInfo info = session->getInfo(popart_id);
    dtype = info.dataType();
  }

#define DEFINE_CASE(value)                                                     \
  case popart::DataType::value: {                                              \
    return PopartType::value;                                                  \
  }

  switch (dtype) { FOR_ALL_POPART_TYPES(DEFINE_CASE) }
#undef DEFINE_CASE

  ERROR("Unsupported popart type in return: " << dtype);
}

void CompilerImpl::cachePopartTypes() {
  for (size_t idx = 1; idx < ids.size(); idx++) {
    ids_types.push_back(getPopartType(idx));
  }
}

void CompilerImpl::errorOnCycleLogging() const {
  ERROR_ON_MSG(popart_options.instrumentWithHardwareCycleCounter,
               "Cycle count logging is only supported on actual IPU hardware.");
}

} // namespace detail
} // namespace popart_compiler
} // namespace poptorch
