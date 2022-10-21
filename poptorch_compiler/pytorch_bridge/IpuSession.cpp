// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "pytorch_bridge/IpuSession.hpp"

#include <llvm/ADT/STLExtras.h>

#include <chrono>
#include <thread>

#include "lower_to_poplar/PopitExecutor.hpp"
#include "lower_to_poplar/PopitSession.hpp"

#include <poplar/Type.hpp>

namespace poptorch_ir {

namespace {

std::size_t dataSize(Type element_type) {
  switch (element_type) {
  case Type::BOOL:
  case Type::CHAR:
  case Type::UNSIGNED_CHAR:
    return 1;
  case Type::SHORT:
  case Type::UNSIGNED_SHORT:
  case Type::HALF:
  case Type::BFLOAT16:
    return 2;
  case Type::INT:
  case Type::UNSIGNED_INT:
  case Type::FLOAT:
    return 4;
  case Type::NONE:
  case Type::UNDEFINED:
    break;
  }
  ERROR("No type");
}

class StaticIpuSession : public IIpuSession {
public:
  Buffer allocate(const TensorType &type) override {
    auto data_size = dataSize(type.element_type) * type.getNumElements();
    return Buffer(std::make_shared<std::vector<char>>(data_size));
  }
  void copyDataFromCpuSource(Buffer &ipu_dest, const char *cpu_data) override {
    const auto &ipu_data = ipu_dest.getCpuData();
    ERROR_ON(!ipu_data);
    std::copy(cpu_data, cpu_data + ipu_data->size(), ipu_data->data());
  }
  void copyDataToCpu(char *cpu_dest, Buffer &ipu_src) override {
    const auto &ipu_data = ipu_src.getCpuData();
    ERROR_ON(!ipu_data);
    std::copy(ipu_data->data(), ipu_data->data() + ipu_data->size(), cpu_dest);
  }
  void copyDataOnDevice(Buffer &dest, const Buffer &src) override {
    const auto &dest_data = dest.getCpuData();
    const auto &src_data = src.getCpuData();
    ERROR_ON(dest_data->size() != src_data->size());
    std::copy(src_data->data(), src_data->data() + src_data->size(),
              dest_data->data());
  }
};

} // namespace

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
