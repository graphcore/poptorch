// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/tensorinfo.hpp>

#include "popart_compiler/CompilerImpl.hpp"
#include "popart_compiler/PopartEnums.hpp"
#include <popart/popx/devicex.hpp>

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

poplar::Type poplarTypeFromPoptorch(poptorch::PopartType type) {
  const popart::DataType popart_type = popartTypeFromPoptorch(type);
  return popart::popx::popType(popart_type);
}

popart::DataType popartTypeFromPoptorch(poptorch::PopartType type) {
  switch (type) {
  case poptorch::PopartType::UINT8:
    return popart::DataType::UINT8;
  case poptorch::PopartType::INT8:
    return popart::DataType::INT8;
  case poptorch::PopartType::UINT16:
    return popart::DataType::UINT16;
  case poptorch::PopartType::INT16:
    return popart::DataType::INT16;
  case poptorch::PopartType::INT32:
    return popart::DataType::INT32;
  case poptorch::PopartType::INT64:
    return popart::DataType::INT64;
  case poptorch::PopartType::UINT32:
    return popart::DataType::UINT32;
  case poptorch::PopartType::UINT64:
    return popart::DataType::UINT64;
  case poptorch::PopartType::BOOL:
    return popart::DataType::BOOL;
  case poptorch::PopartType::FLOAT:
    return popart::DataType::FLOAT;
  case poptorch::PopartType::FLOAT16:
    return popart::DataType::FLOAT16;
  case poptorch::PopartType::BFLOAT16:
    return popart::DataType::BFLOAT16;
  case poptorch::PopartType::DOUBLE:
    return popart::DataType::DOUBLE;
  case poptorch::PopartType::COMPLEX64:
    return popart::DataType::COMPLEX64;
  case poptorch::PopartType::COMPLEX128:
    return popart::DataType::COMPLEX128;
  case poptorch::PopartType::STRING:
    return popart::DataType::STRING;
  case poptorch::PopartType::UNDEFINED:
    return popart::DataType::UNDEFINED;
  default:
    ERROR("Unsupported type in popartTypeFromPoptorchType");
  }

  return popart::DataType::UNDEFINED;
}

} // namespace poptorch
