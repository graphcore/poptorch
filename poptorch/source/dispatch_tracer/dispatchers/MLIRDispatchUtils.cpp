// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "MLIRDispatchUtils.hpp"

#include "poptorch_logging/Error.hpp"

namespace poptorch {

c10::ScalarType compilerTypeToScalarType(poptorch_ir::Type type) {
  switch (type) {
  case poptorch_ir::Type::BOOL:
    return c10::ScalarType::Bool;
  case poptorch_ir::Type::CHAR:
    return c10::ScalarType::Char;
  case poptorch_ir::Type::UNSIGNED_CHAR:
    return c10::ScalarType::Byte;
  case poptorch_ir::Type::SHORT:
    return c10::ScalarType::Short;
  case poptorch_ir::Type::INT:
    return c10::ScalarType::Int;
  case poptorch_ir::Type::HALF:
    return c10::ScalarType::Half;
  case poptorch_ir::Type::FLOAT:
    return c10::ScalarType::Float;
  case poptorch_ir::Type::BFLOAT16:
    return c10::ScalarType::BFloat16;
  default:
    ERROR("No at::scalar_type for the given compiler type");
    return c10::ScalarType::Undefined;
  }
}

} // namespace poptorch
