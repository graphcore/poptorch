// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <model_runtime/DeviceManager.hpp>

#include "lower_to_poplar/PoplarDeviceAndTarget.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {

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

std::shared_ptr<model_runtime::Device> getDefaultDevice() {
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
    using WaitStrategy = model_runtime::DeviceWaitStrategy;
    model_runtime::DeviceWaitConfig wait_config;
    wait_config.strategy = waitIfIpuIsUnavailable() ? WaitStrategy::WAIT_FOREVER
                                                    : WaitStrategy::NO_WAIT;
    device = manager.getDevice(1, {}, wait_config);
  }
  ERROR_ON_MSG(!device, "Failed to acquire a device");
  return device;
}
} // namespace

PoplarTarget::PoplarTarget(const poplar::Target &target)
    : _target(new poplar::Target(target)) {}

PoplarTarget::~PoplarTarget() = default;

const poplar::Target &PoplarTarget::target() const { return *_target; }

PoplarDevice::PoplarDevice(std::shared_ptr<model_runtime::Device> device)
    : _device(std::move(device)) {}

PoplarTarget PoplarDevice::getTarget() const {
  return PoplarTarget(_device->device().getTarget());
}

PoplarDevice::~PoplarDevice() = default;

const poplar::Device &PoplarDevice::device() const { return _device->device(); }

poplar::Device &PoplarDevice::device() { return _device->device(); }

PoplarDevice PoplarDevice::defaultDevice() {
  return PoplarDevice(getDefaultDevice());
}

} // namespace poptorch_ir
