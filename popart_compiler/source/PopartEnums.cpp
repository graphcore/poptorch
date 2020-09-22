// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

//#include <popart/onnxutil.hpp>
#include <popart/tensorinfo.hpp>

#include "popart_compiler/PopartEnums.hpp"

namespace popart {
namespace onnxutil {
DataType getDataType(int);
} // namespace onnxutil
} // namespace popart

namespace poptorch {

const char *onnxStrFromDtypeInt(int64_t dtype) {
  auto popart_type = popart::onnxutil::getDataType(dtype);
  const auto &data_type_map(popart::getDataTypeInfoMap());

  // data_type_map is static so the c_str() remains valid
  return data_type_map.at(popart_type).name().c_str();
}

} // namespace poptorch
