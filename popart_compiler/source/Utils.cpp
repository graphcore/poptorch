// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <thread>

#include <popart/popx/devicex.hpp>
#include <popart/popx/devicexmanager.hpp>
#include <popart/tensorinfo.hpp>

#include "popart_compiler/CompilerImpl.hpp"
#include "popart_compiler/PopartEnums.hpp"
#include "popart_compiler/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

// These symbols exist in popart but are not declared publicly
namespace ONNX_NAMESPACE {
enum class TensorProto_DataType;
} // namespace ONNX_NAMESPACE

namespace popart {
namespace onnxutil {
DataType getDataType(int);
ONNX_NAMESPACE::TensorProto_DataType getTPDataType(DataType data_type);
} // namespace onnxutil
} // namespace popart

namespace poptorch {
namespace popart_compiler {

bool ipuModelEnvironmentVariableIsEnabled() {
  if (const char *env_use_model = std::getenv("POPTORCH_IPU_MODEL")) {
    const bool model_enabled = std::stoi(env_use_model) != 0;
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
    const bool model_enabled = std::stoi(env_use_model) != 0;
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
  return "ipu2"; // Default to MK2 if unspecified
}

int getNumTilesPerIpu(const std::string &ipu_model_version) {
  int num_tiles_per_ipu = 0;

  if (ipu_model_version == "ipu1") {
    num_tiles_per_ipu = 1216; // MK1
  }
  if (ipu_model_version == "ipu2") {
    num_tiles_per_ipu = 1472; // MK2
  }
  if (ipu_model_version == "ipu21") {
    num_tiles_per_ipu = 1472; // C600
  }

  if (ipuSmallModelEnvironmentVariableIsEnabled()) {
    num_tiles_per_ipu = 4;
  }

  ERROR_ON_MSG((ipu_model_version.find("ipu:") == std::string::npos) &&
                   (num_tiles_per_ipu == 0),
               "Invalid IPU model version. Valid versions: ipu1, ipu2, ipu21.");
  return num_tiles_per_ipu;
}

// Round up the number of IPUs, if required, to the minimum number which need
// to be reservered
std::uint64_t roundUpNumIPUs(std::uint64_t num_ipus) {
  std::uint64_t rounded_num_ipus = 1;

  // If fewer than 64, find the next power of 2
  while (rounded_num_ipus < num_ipus) {
    rounded_num_ipus *= 2;
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

bool waitForAWhile() {
  constexpr std::int64_t sleep_time = 15;
  logging::trace("No IPU available, sleeping for {} seconds", sleep_time);
  std::this_thread::sleep_for(std::chrono::seconds(sleep_time));
  return true;
}

std::int64_t ipuHardwareVersion(std::uint64_t num_ipus) {
  if (ipuModelEnvironmentVariableIsEnabled() ||
      ipuSmallModelEnvironmentVariableIsEnabled()) {
    return 0;
  }
  auto devices = popart::DeviceManager::createDeviceManager().enumerateDevices(
      popart::SyncPattern::Full, num_ipus);
  if (devices.empty()) {
    return 0;
  }
  const std::string arch = devices.front()->getTarget().getTargetArchString();

  // The architecture string must be 'ipu' followed by one or more non-zero
  // digits.
  bool is_valid = arch.size() > 3 && arch.find("ipu", 0) == 0;
  for (size_t i = 3; is_valid && i < arch.size(); ++i) {
    is_valid = arch[i] > '0' && arch[i] <= '9';
  }

  if (!is_valid) {
    logging::warn("Unknown IPU version: {} (Expected 'ipuX' "
                  " where X is one or more strictly positive digits)",
                  arch);
    return -1;
  }
  return std::atoi(arch.substr(3).c_str());
}

std::unique_ptr<char[]> stringToUniquePtr(const std::string &str) {
  auto ptr = std::unique_ptr<char[]>(new char[str.size() + 1]);
  str.copy(ptr.get(), std::string::npos);
  ptr.get()[str.size()] = '\0';
  return ptr;
}

int64_t dtypeIntFromOnnxStr(const char *onnx_type) {
  auto popart_type = popart::dataTypeFromString(onnx_type);
  return static_cast<int64_t>(popart::onnxutil::getTPDataType(popart_type));
}

const char *onnxStrFromDtypeInt(int64_t dtype) {
  auto popart_type = popart::onnxutil::getDataType(dtype);
  const auto &data_type_map(popart::getDataTypeInfoMap());

  // data_type_map is static so the c_str() remains valid
  return data_type_map.at(popart_type).name().c_str();
}

poplar::Type poplarTypeFromPoptorch(PopartType type) {
  const popart::DataType popart_type = popartTypeFromPoptorch(type);
  return popart::popx::popType(popart_type);
}

popart::DataType popartTypeFromPoptorch(PopartType type) {
  switch (type) {
  case PopartType::UINT8:
    return popart::DataType::UINT8;
  case PopartType::INT8:
    return popart::DataType::INT8;
  case PopartType::UINT16:
    return popart::DataType::UINT16;
  case PopartType::INT16:
    return popart::DataType::INT16;
  case PopartType::INT32:
    return popart::DataType::INT32;
  case PopartType::INT64:
    return popart::DataType::INT64;
  case PopartType::UINT32:
    return popart::DataType::UINT32;
  case PopartType::UINT64:
    return popart::DataType::UINT64;
  case PopartType::BOOL:
    return popart::DataType::BOOL;
  case PopartType::FLOAT:
    return popart::DataType::FLOAT;
  case PopartType::FLOAT16:
    return popart::DataType::FLOAT16;
  case PopartType::BFLOAT16:
    return popart::DataType::BFLOAT16;
  case PopartType::FLOAT8_143:
    return popart::DataType::FLOAT8_143;
  case PopartType::FLOAT8_152:
    return popart::DataType::FLOAT8_152;
  case PopartType::DOUBLE:
    return popart::DataType::DOUBLE;
  case PopartType::COMPLEX64:
    return popart::DataType::COMPLEX64;
  case PopartType::COMPLEX128:
    return popart::DataType::COMPLEX128;
  case PopartType::STRING:
    return popart::DataType::STRING;
  case PopartType::UNDEFINED:
    return popart::DataType::UNDEFINED;
  default:
    ERROR("Unsupported type in popartTypeFromPoptorchType");
  }

  return popart::DataType::UNDEFINED;
}
} // namespace popart_compiler
} // namespace poptorch
