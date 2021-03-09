// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <thread>

#include <popart/popx/devicexmanager.hpp>

#include "popart_compiler/Utils.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch {

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

  if (ipuSmallModelEnvironmentVariableIsEnabled()) {
    num_tiles_per_ipu = 4;
  }

  ERROR_ON_MSG(num_tiles_per_ipu == 0,
               "Invalid IPU model version. Valid versions: ipu1, ipu2.");
  return num_tiles_per_ipu;
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
  if (arch.size() != 4 || arch.rfind("ipu", 0) != 0 || arch[3] < '1' ||
      arch[3] > '9') {
    logging::warn("Unknown IPU version: {} (Expected 4 characters string "
                  "'ipuX' where X is a digit > 0)",
                  arch);
    return -1;
  }
  return arch[3] - '0';
}

std::unique_ptr<char[]> stringToUniquePtr(const std::string &str) {
  auto ptr = std::unique_ptr<char[]>(new char[str.size() + 1]);
  str.copy(ptr.get(), std::string::npos);
  ptr.get()[str.size()] = '\0';
  return ptr;
}

} // namespace poptorch
