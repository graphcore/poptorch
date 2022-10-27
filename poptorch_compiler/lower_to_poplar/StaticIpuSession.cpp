// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/StaticIpuSession.hpp"

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

} // namespace

Buffer StaticIpuSession::allocate(const TensorType &type) {
  auto data_size = dataSize(type.element_type) * type.getNumElements();
  return Buffer(std::make_shared<std::vector<char>>(data_size));
}

void StaticIpuSession::copyDataFromCpuSource(Buffer &ipu_dest,
                                             const char *cpu_data) {
  const auto &ipu_data = ipu_dest.getCpuData();
  ERROR_ON(!ipu_data);
  std::copy(cpu_data, cpu_data + ipu_data->size(), ipu_data->data());
}

void StaticIpuSession::copyDataToCpu(char *cpu_dest, Buffer &ipu_src) {
  const auto &ipu_data = ipu_src.getCpuData();
  ERROR_ON(!ipu_data);
  std::copy(ipu_data->data(), ipu_data->data() + ipu_data->size(), cpu_dest);
}

void StaticIpuSession::copyDataOnDevice(Buffer &dest, const Buffer &src) {
  const auto &dest_data = dest.getCpuData();
  const auto &src_data = src.getCpuData();
  ERROR_ON(dest_data->size() != src_data->size());
  std::copy(src_data->data(), src_data->data() + src_data->size(),
            dest_data->data());
}

} // namespace poptorch_ir
