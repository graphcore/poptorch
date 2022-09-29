// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_TYPES_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_COMPILER_TYPES_HPP_

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

namespace poptorch_ir {

// Host blob of memory containing data to transfer to the IPU.
using Buffer = std::shared_ptr<std::vector<char>>;

// A token representing an SSA value on our side. PyTorch records it's
// tensors->TensorId and we record TensorId->mlir::Value. This stops either side
// from depending directly on each others internal representation.
using TensorId = std::uint32_t;

// This is identical except that it is known to be valid for it to be none_id
using OptionalTensorId = std::uint32_t;

// So we can signal that a tensor was invalid (Just for so unimplemented
// functions can return something right now.)
constexpr TensorId tensor_error_id = std::numeric_limits<TensorId>::max();

// The tensor is none (e.g. optional parameter/return) and this is not an error
constexpr TensorId none_id = std::numeric_limits<TensorId>::max() - 1;

// How to calculate which floating-point outputs require gradients (others
// types will always have this set to false.)
enum class RequiresGradType {
  OR_INPUTS, // OR together all the input tensor requires_grad values
  FALSE      // always false
};

struct ODSTensorResult {
  std::vector<TensorId> tensor_ids;
  std::vector<RequiresGradType> requires_grad_types;
};

// When returning an MLIR op, each return could be compulsory, optional or
// variadic tensor under the MLIR Operation Definition Specification (ODS).
// Using a vector for each return allows each return to be optional or variadic.
using ODSTensorResults = std::vector<ODSTensorResult>;

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
  BFLOAT16,
  NONE,
  UNDEFINED,
};

struct TensorType {
  std::vector<int64_t> shape;
  poptorch_ir::Type element_type;

  std::int64_t getNumElements() const {
    return std::accumulate(shape.begin(), shape.end(), std::int64_t{1},
                           std::multiplies<>());
  }
};

struct StreamInfo {
  std::vector<char> name;
  Buffer buff;

  TensorType type;
};

} // namespace poptorch_ir

#endif
