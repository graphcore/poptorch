// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/EagerIpuSession.hpp"

#include <chrono>
#include <memory>
#include <thread>

#include <popit/Device.hpp>
#include <popit/popit.hpp>

#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {

namespace {

poplar::Type poplarType(Type element_type) {
  switch (element_type) {
  case Type::BOOL:
    return poplar::BOOL;
  case Type::CHAR:
    return poplar::CHAR;
  case Type::UNSIGNED_CHAR:
    return poplar::UNSIGNED_CHAR;
  case Type::SHORT:
    return poplar::SHORT;
  case Type::UNSIGNED_SHORT:
    return poplar::UNSIGNED_SHORT;
  case Type::HALF:
  case Type::BFLOAT16:
    return poplar::HALF;
  case Type::INT:
    return poplar::INT;
  case Type::UNSIGNED_INT:
    return poplar::UNSIGNED_INT;
  case Type::FLOAT:
    return poplar::FLOAT;
  case Type::NONE:
  case Type::UNDEFINED:
    break;
  }
  ERROR("No type");
}

auto allocatePopitMemory(const std::shared_ptr<popit::Session_t> &session,
                         const TensorType &type) {
  const auto destructor = [weak_session =
                               std::weak_ptr(session)](popit::Mem_t *ptr) {
    // If the session has already been destroyed it will have freed all its
    // memory so we no longer need to free this pointer
    if (const auto locked_session = weak_session.lock()) {
      popit::free(ptr);
    }
  };
  return std::shared_ptr<popit::Mem_t>(
      popit::malloc(session.get(), poplarType(type.element_type),
                    type.getNumElements()),
      destructor);
}

bool shouldWaitIfIpuIsUnavailable() {
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

std::unique_ptr<popit::Device> getPopitDevice() {
  bool model_enabled = false;
  if (const char *env_use_model = std::getenv("POPTORCH_IPU_MODEL")) {
    model_enabled = std::stoi(env_use_model) != 0;
    ERROR_ON_MSG(model_enabled, "IPU model is unsupported in eager mode");
  }
  if (const char *env_use_model = std::getenv("POPTORCH_SMALL_IPU_MODEL")) {
    model_enabled = std::stoi(env_use_model) != 0;
    ERROR_ON_MSG(model_enabled, "IPU model is unsupported in eager mode");
  }

  // Otherwise attempt to acquire hardware
  const popit::DeviceManager device_manager;
  auto devices = device_manager.getDevices(/*requiredNumIpus=*/1);
  if (devices.empty()) {
    ERROR("No devices found");
  }

  const bool wait_for_ipu = shouldWaitIfIpuIsUnavailable();
  do {
    for (auto &device : devices) {
      if (device.attach()) {
        return std::make_unique<popit::Device>(std::move(device));
      }
    }
    if (wait_for_ipu) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  } while (wait_for_ipu);
  ERROR("Failed to attach to any of the IPU devices.");
  return nullptr;
}

auto createSession(popit::Device &device) {
  const poplar::Target target = device.getTarget();

  auto session = std::shared_ptr<popit::Session_t>(
      popit::createSession(&target),
      [](popit::Session *s) { popit::destroySession(s); });

  popit::connect(session.get(), &device);
  return session;
}

} // namespace

EagerIpuSession::EagerIpuSession()
    : device(getPopitDevice()), session(createSession(*device)) {}

EagerIpuSession::~EagerIpuSession() {
  // Ensure that the device outlives the session
  session.reset();
}

Buffer EagerIpuSession::allocate(const TensorType &type) {
  return Buffer(PopitMemPtr(allocatePopitMemory(session, type)));
}

void EagerIpuSession::copyDataFromCpuSource(Buffer &ipu_dest,
                                            const char *cpu_data) {
  popit::copyFromHost(cpu_data, ipu_dest.getPopitData().get());
}

void EagerIpuSession::copyDataToCpu(char *cpu_dest, Buffer &ipu_src) {
  popit::copyToHost(ipu_src.getPopitData().get(), cpu_dest);
}

void EagerIpuSession::copyDataOnDevice(Buffer &dest, const Buffer &src) {
  popit::copy(src.getPopitData().get(), dest.getPopitData().get());
}

} // namespace poptorch_ir
