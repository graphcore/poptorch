// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "pytorch_bridge/IpuSession.hpp"

#include <llvm/ADT/STLExtras.h>

#include <chrono>
#include <memory>
#include <thread>
#include <utility>

#include "pytorch_bridge/DebugInfo.hpp"

#include <poptorch_logging/Logging.hpp>

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
Buffer::Buffer(CpuBuffer buf) noexcept : _store(std::move(buf)) {}
Buffer &Buffer::operator=(CpuBuffer buf) noexcept {
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
bool Buffer::hasData() const {
  return !std::holds_alternative<std::monostate>(_store);
}

std::shared_ptr<IIpuSession> createStaticSession() {
  return std::make_shared<StaticIpuSession>();
}
} // namespace poptorch_ir
