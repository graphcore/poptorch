// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "pytorch_bridge/IpuSession.hpp"

#include <llvm/ADT/STLExtras.h>

#include <chrono>
#include <thread>

#include "lower_to_poplar/EagerIpuSession.hpp"
#include "lower_to_poplar/PopitExecutor.hpp"
#include "lower_to_poplar/StaticIpuSession.hpp"

namespace poptorch_ir {

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
std::shared_ptr<IIpuSession> createEagerSession() {
  return std::make_shared<EagerIpuSession>();
}

PopitDeviceFunctionWrapper::PopitDeviceFunctionWrapper(
    std::unique_ptr<PopitDeviceFunction> func)
    : _func(std::move(func)) {}
PopitDeviceFunctionWrapper::~PopitDeviceFunctionWrapper() = default;

void PopitDeviceFunctionWrapper::run(IAllocationMap &alloc_map) const {
  std::vector<popit::Mem_t *> inputs;
  inputs.reserve(_func->getInputs().size());
  llvm::transform(
      _func->getInputs(), std::back_inserter(inputs),
      [&](TensorId input) { return alloc_map.getAllocation(input); });

  std::vector<popit::Mem_t *> outputs;
  outputs.reserve(_func->getOutputs().size());
  llvm::transform(
      _func->getOutputs(), std::back_inserter(outputs),
      [&](TensorId output) { return alloc_map.getOrAllocate(output); });

  _func->run(inputs, outputs);
}

} // namespace poptorch_ir
