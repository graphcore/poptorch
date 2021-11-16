// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_TYPES_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_TYPES_HPP_

#include <cstdint>
#include <limits>

namespace poptorch_ir {

// A token representing an SSA value on our side. PyTorch records it's
// tensors->TensorId and we record TensorId->mlir::Value. This stops either side
// from depending directly on each others internal representation.
using TensorId = std::uint32_t;

// So we can signal that a tensor was invalid (Just for so unimplemented
// functions can return something right now.)
constexpr TensorId tensor_error_id = std::numeric_limits<TensorId>::max();

enum class Type : std::uint8_t {
  BOOL,
  CHAR,
  UNSIGNED_CHAR,
  SHORT,
  UNSIGNED_SHORT,
  INT,
  UNSIGNED_INT,
  HALF,
  FLOAT,
};

} // namespace poptorch_ir

#endif
