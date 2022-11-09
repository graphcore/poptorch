// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "pytorch_bridge/IpuSession.hpp"

#include <llvm/ADT/STLExtras.h>

#include <chrono>
#include <memory>
#include <thread>
#include <utility>

#include "lower_to_poplar/EagerIpuSession.hpp"
#include "lower_to_poplar/PopitExecutor.hpp"
#include "lower_to_poplar/StaticIpuSession.hpp"
#include "pytorch_bridge/DebugInfo.hpp"

#include <poptorch_logging/Logging.hpp>

namespace poptorch_ir {

PopitMemPtr::PopitMemPtr(std::nullptr_t)
    : std::shared_ptr<popit::Mem_t>(nullptr) {}
PopitMemPtr::PopitMemPtr(std::shared_ptr<popit::Mem_t> ptr)
    : std::shared_ptr<popit::Mem_t>(std::move(ptr)) {}

Buffer::Buffer(CpuBuffer buf) noexcept : _store(std::move(buf)) {}
Buffer::Buffer(PopitMemPtr buf) noexcept : _store(std::move(buf)) {}
Buffer &Buffer::operator=(CpuBuffer buf) noexcept {
  _store = std::move(buf);
  return *this;
}
Buffer &Buffer::operator=(PopitMemPtr buf) noexcept {
  _store = std::move(buf);
  return *this;
}
const CpuBuffer &Buffer::getCpuData() {
  ERROR_ON(!std::holds_alternative<CpuBuffer>(_store));
  return std::get<CpuBuffer>(_store);
}
const CpuBuffer &Buffer::getCpuData() const {
  ERROR_ON(!std::holds_alternative<CpuBuffer>(_store));
  return std::get<CpuBuffer>(_store);
}
PopitMemPtr &Buffer::getPopitData() {
  ERROR_ON(!std::holds_alternative<PopitMemPtr>(_store));
  return std::get<PopitMemPtr>(_store);
}
const PopitMemPtr &Buffer::getPopitData() const {
  ERROR_ON(!std::holds_alternative<PopitMemPtr>(_store));
  return std::get<PopitMemPtr>(_store);
}
bool Buffer::hasData() const {
  return !std::holds_alternative<std::monostate>(_store);
}

std::shared_ptr<IIpuSession> createStaticSession() {
  return std::make_shared<StaticIpuSession>();
}

std::shared_ptr<IIpuSession> createEagerSession(bool headless) {
  if (headless) {
    return std::make_shared<HeadlessIpuSession>();
  }
  return std::make_shared<EagerIpuSession>();
}

PopitDeviceFunctionWrapper
PopitDeviceFunctionWrapper::createTrivialFunction() noexcept {
  return PopitDeviceFunctionWrapper(nullptr, {}, {});
}

PopitDeviceFunctionWrapper::PopitDeviceFunctionWrapper(
    std::shared_ptr<PopitDeviceFunction> func, FunctionIO io,
    GraphDebugInfo debug_info)
    : _func(std::move(func)), _io(std::move(io)),
      _debug_info(std::move(debug_info)) {}
PopitDeviceFunctionWrapper::~PopitDeviceFunctionWrapper() = default;

void PopitDeviceFunctionWrapper::run(IAllocationMap &alloc_map) const {
  ERROR_ON(_func == nullptr);

  std::vector<popit::Mem_t *> inputs;
  inputs.reserve(_io.inputs.size());
  llvm::transform(_io.inputs, std::back_inserter(inputs), [&](TensorId input) {
    return alloc_map.getAllocation(input);
  });

  std::vector<popit::Mem_t *> outputs;
  outputs.reserve(_io.outputs.size());
  llvm::transform(llvm::enumerate(_io.outputs), std::back_inserter(outputs),
                  [&](auto output) {
                    return alloc_map.getOrAllocate(
                        output.value(),
                        TensorDebugInfo{_debug_info, output.index()});
                  });

  _func->run(inputs, outputs);

  poptorch::logging::info("Executed PopIT function");
}

bool PopitDeviceFunctionWrapper::isTrivial() const noexcept {
  return _func == nullptr;
}

} // namespace poptorch_ir
