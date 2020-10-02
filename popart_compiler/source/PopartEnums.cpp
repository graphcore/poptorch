// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/tensorinfo.hpp>

#include "popart_compiler/PopartEnums.hpp"

namespace ONNX_NAMESPACE {
enum class TensorProto_DataType;
} // namespace ONNX_NAMESPACE

namespace popart {
namespace onnxutil {
DataType getDataType(int);
ONNX_NAMESPACE::TensorProto_DataType getTPDataType(DataType data_type);
} // namespace onnxutil
} // namespace popart

namespace poptorch {

int64_t dtypeIntFromOnnxStr(const char *onnx_type) {
  auto popart_type = popart::dataTypeFromString(onnx_type);
  return static_cast<int64_t>(popart::onnxutil::getTPDataType(popart_type));
}

const char *onnxStrFromDtypeInt(int64_t dtype) {
  auto popart_type = popart::onnxutil::getDataType(dtype);
  const auto &data_type_map(popart::getDataTypeInfoMap());

  // data_type_map is static so the c_str() remains valid
  return data_type_map.at(popart_type).name().c_str();
}

} // namespace poptorch
