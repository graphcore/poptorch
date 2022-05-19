// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <model_runtime/DeviceManager.hpp>

#include "lower_to_poplar/CompilerHelpers.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {

poplar::Type elementTypeFromMLIR(mlir::Type elementType) {
  if (elementType.isF16()) {
    return poplar::HALF;
  }
  if (elementType.isF32()) {
    return poplar::FLOAT;
  }
  if (elementType.isUnsignedInteger(8)) {
    return poplar::UNSIGNED_CHAR;
  }
  if (elementType.isUnsignedInteger(16)) {
    return poplar::UNSIGNED_SHORT;
  }
  if (elementType.isUnsignedInteger(32) || elementType.isUnsignedInteger(64)) {
    return poplar::UNSIGNED_INT;
  }
  // We use isInteger from here onwards to capture both
  // isSignedInteger and isSignlessInteger
  if (elementType.isInteger(1)) {
    return poplar::BOOL;
  }
  if (elementType.isInteger(8)) {
    return poplar::SIGNED_CHAR;
  }
  if (elementType.isInteger(16)) {
    return poplar::SHORT;
  }
  if (elementType.isInteger(32) || elementType.isInteger(64)) {
    return poplar::INT;
  }
  ERROR("Unsupported MLIR type");

  return poplar::FLOAT;
}

namespace {
bool waitIfIpuIsUnavailable() {
  bool wait = false;
  if (const char *env_wait_for_ipu = std::getenv("POPTORCH_WAIT_FOR_IPU")) {
    wait = std::stoi(env_wait_for_ipu) != 0;
    poptorch::logging::info(
        "From POPTORCH_WAIT_FOR_IPU environment variable: If no IPU "
        "is available: {}",
        wait ? "Wait" : "Fail & exit");
  }
  return wait;
}
} // namespace

std::shared_ptr<model_runtime::Device> getDevice() {
  std::shared_ptr<model_runtime::Device> device;
  model_runtime::DeviceManager manager;

  bool model_enabled = false;

  // Run on model if the env var is set.
  if (const char *env_use_model = std::getenv("POPTORCH_IPU_MODEL")) {
    model_enabled = std::stoi(env_use_model) != 0;
    if (model_enabled) {
      device = manager.createIpuModelDevice(1);
    }
  }
  if (const char *env_use_model = std::getenv("POPTORCH_SMALL_IPU_MODEL")) {
    model_enabled = std::stoi(env_use_model) != 0;
    if (!device && model_enabled) {
      device = manager.createSmallIpuModelDevice(1);
    }
  }
  // Run on an actual device.
  if (!device) {
    device =
        manager.getDevice(1, {},
                          waitIfIpuIsUnavailable() ? model_runtime::WAIT_FOREVER
                                                   : model_runtime::NO_WAIT);
  }
  ERROR_ON_MSG(!device, "Failed to acquire a device");
  return device;
}
} // namespace poptorch_ir
