// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/EagerIpuSession.hpp"

#include <chrono>
#include <memory>
#include <thread>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>

#include <popit/Device.hpp>
#include <popit/popit.hpp>

#include "dialect/Helpers.hpp"
#include "lower_to_poplar/NonRestartingMLIRTimer.hpp"
#include "lower_to_poplar/PopitExecutor.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"
#include "pytorch_bridge/IpuSession.hpp"

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

std::string getIpuModelVersion() {
  if (const char *env_ipu_model_version =
          std::getenv("POPTORCH_IPU_MODEL_VERSION")) {
    return env_ipu_model_version;
  }
  return "ipu2"; // Default to MK2 if unspecified
}

std::unique_ptr<popit::Device> getPopitDevice() {
  bool model_enabled = false;
  if (const char *env_use_model = std::getenv("POPTORCH_IPU_MODEL")) {
    model_enabled = std::stoi(env_use_model) != 0;
    if (model_enabled) {
      return std::make_unique<popit::Device>(
          popit::Device::createModelDevice(getIpuModelVersion()));
    }
  }
  if (const char *env_use_model = std::getenv("POPTORCH_SMALL_IPU_MODEL")) {
    model_enabled = std::stoi(env_use_model) != 0;
    if (model_enabled) {
      return std::make_unique<popit::Device>(
          popit::Device::createModelDevice(getIpuModelVersion(), 1, 4));
    }
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

llvm::hash_code
hashOp(mlir::Operation *op,
       const llvm::DenseMap<mlir::Value, llvm::hash_code> &tensor_hash_map) {
  auto hash = llvm::hash_value(op->getName().getStringRef());

  // Hash attributes.
  const auto &dict = op->getAttrDictionary();
  for (const auto *it = dict.begin(); it != dict.end(); it++) {
    hash = llvm::hash_combine(hash, it->getName().strref(),
                              mlirToStr(it->getValue()));
  }

  // Combine the pre-computed hash of the op's inputs.
  for (const auto &operand : op->getOperands()) {
    auto it = tensor_hash_map.find(operand);
    ERROR_ON(it == tensor_hash_map.end());
    hash = llvm::hash_combine(hash, it->second);
  }

  // Hash the op's output types.
  for (const auto &type : op->getResultTypes()) {
    hash = llvm::hash_combine(hash, mlirToStr(type));
  }

  return hash;
}

llvm::hash_code hashGraph(mlir::ModuleOp module) {
  llvm::hash_code hash{0};

  // Build up hashes for each of the values as they are traced through the
  // parent function.
  llvm::DenseMap<mlir::Value, llvm::hash_code> tensor_hash_map;

  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    tensor_hash_map.clear();
    for (const auto &argument : func.getArguments()) {
      tensor_hash_map.insert(
          {argument, llvm::hash_value(argument.getArgNumber())});
    }

    func->walk([&](mlir::Operation *op) {
      if (op == func) {
        return;
      }

      if (mlir::dyn_cast_or_null<mlir::func::ReturnOp>(op) != nullptr) {
        for (const auto &operand : op->getOperands()) {
          auto it = tensor_hash_map.find(operand);
          ERROR_ON(it == tensor_hash_map.end());
          hash = llvm::hash_combine(hash, it->second);
        }

        return;
      }

      auto meoi = mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(op);
      // All ops must have their memory side effects captured by a MemoryEffect
      // trait, otherwise, we will not be able to identify any side effects.
      ERROR_ON(meoi == nullptr);
      // Memory read side effects are not yet supported.
      ERROR_ON(meoi.hasEffect<mlir::MemoryEffects::Read>());
      auto op_hash = hashOp(op, tensor_hash_map);
      if (meoi.hasEffect<mlir::MemoryEffects::Write>()) {
        // If there is a write side effect, update the global hash right away.
        hash = llvm::hash_combine(hash, op_hash);
      }

      // Update the tensor map by creating a hash for every result based on the
      // op's hash and the index of the result.
      for (const auto &result : op->getResults()) {
        tensor_hash_map.insert(
            {result, llvm::hash_combine(
                         op_hash, llvm::hash_value(result.getResultNumber()))});
      }
    });
  }

  poptorch::logging::trace("Graph hash is {}", hash);
  return hash;
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

std::shared_ptr<PopitDeviceFunction>
PopitFunctionCache::emplace(const mlir::ModuleOp &graph,
                            EagerIpuSession &session,
                            NonRestartingMLIRTimer &timer) {
  auto hash = hashGraph(graph);

  if (auto it = _cache.find(hash); it != _cache.end()) {
    poptorch::logging::trace("Found graph in cache: reusing PopIT function");
    return it->second;
  }
  poptorch::logging::trace("Graph not cached: compiling new PopIT function");
  auto func = std::make_shared<PopitDeviceFunction>(session, graph, timer);
  _cache.insert({hash, func});
  return func;
}

PopitDeviceFunctionWrapper PopitFunctionCache::emplaceWrapped(
    const mlir::ModuleOp &graph, EagerIpuSession &session, FunctionIO io,
    GraphDebugInfo debug_info, NonRestartingMLIRTimer &timer) {
  return PopitDeviceFunctionWrapper(emplace(graph, session, timer),
                                    std::move(io), std::move(debug_info));
}

PopitDeviceFunctionWrapper
EagerIpuSession::createFunction(const mlir::ModuleOp &graph, FunctionIO io,
                                GraphDebugInfo debug_info,
                                NonRestartingMLIRTimer &timer) {
  return _func_cache.emplaceWrapped(graph, *this, std::move(io),
                                    std::move(debug_info), timer);
}

Buffer HeadlessIpuSession::allocate(const TensorType &type) {
  UNUSED(type);
  // We must return a non-null value from allocate. In order to better test the
  // providence of the buffers we pass a pointer to the current
  // HeadlessIpuSession. It is safe to do this because the only way to access
  // the buffer is through this class
  return Buffer(PopitMemPtr(std::shared_ptr<popit::Mem_t>(
      reinterpret_cast<popit::Mem_t *>(this), [](popit::Mem_t *) {})));
}

void HeadlessIpuSession::copyDataFromCpuSource(Buffer &ipu_dest,
                                               const char *cpu_data) {
  ERROR_ON_MSG(reinterpret_cast<HeadlessIpuSession *>(
                   ipu_dest.getPopitData().get()) != this,
               "copyDataFromCpuSource: the dest buffer does not belong to this "
               "HeadlessIpuSession");
  ERROR_ON(!cpu_data);
}

void HeadlessIpuSession::copyDataToCpu(char *cpu_dest, Buffer &ipu_src) {
  ERROR_ON_MSG(reinterpret_cast<HeadlessIpuSession *>(
                   ipu_src.getPopitData().get()) != this,
               "copyDataToCpu: the source buffer does not belong to this "
               "HeadlessIpuSession");
  ERROR_ON(!cpu_dest);
}

void HeadlessIpuSession::copyDataOnDevice(Buffer &dest, const Buffer &src) {
  ERROR_ON_MSG(
      reinterpret_cast<HeadlessIpuSession *>(dest.getPopitData().get()) != this,
      "copyDataOnDevice: the dest buffer does not belong to this "
      "HeadlessIpuSession");
  ERROR_ON_MSG(
      reinterpret_cast<HeadlessIpuSession *>(src.getPopitData().get()) != this,
      "copyDataOnDevice: the source buffer does not belong to this "
      "HeadlessIpuSession");
}

PopitDeviceFunctionWrapper
HeadlessIpuSession::createFunction(const mlir::ModuleOp &graph, FunctionIO io,
                                   GraphDebugInfo debug_info,
                                   NonRestartingMLIRTimer &timer) {
  UNUSED(graph);
  UNUSED(timer);
  return PopitDeviceFunctionWrapper(
      std::make_shared<PopitDeviceFunction>(PopitDeviceFunction()),
      std::move(io), std::move(debug_info));
}

} // namespace poptorch_ir
