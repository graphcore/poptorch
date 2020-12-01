// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_COMPILER_POPART_ENUMS_HPP
#define POPART_COMPILER_POPART_ENUMS_HPP
#include <string>

#include "poptorch_logging/Error.hpp"

namespace poptorch {

/*
 * We maintain an ABI boundary inbetween PopART and Torch JIT. This avoids the
 * issue of torch being compiled with different CXX_ABI versions. However it
 * means we must replicate PopART enums here so they can be shared by both.
 */

// The training optimizer algorithm used.
enum class OptimizerType : std::uint8_t {
  SGD = 0,
  ADAMW,
  ADAMW_NO_BIAS,
  RMSPROP,
  RMSPROP_CENTERED,
  LAMB,
  LAMB_NO_BIAS,
  NONE
};

#define FOR_ALL_FIXED_POINT_TYPES(_)                                           \
  _(UINT8)                                                                     \
  _(INT8)                                                                      \
  _(UINT16)                                                                    \
  _(INT16)                                                                     \
  _(INT32)                                                                     \
  _(INT64)                                                                     \
  _(UINT32)                                                                    \
  _(UINT64)                                                                    \
  _(BOOL)

#define FOR_ALL_FLOATING_POINT_TYPES(_)                                        \
  _(FLOAT)                                                                     \
  _(FLOAT16)                                                                   \
  _(BFLOAT16)                                                                  \
  _(DOUBLE)                                                                    \
  _(COMPLEX64)                                                                 \
  _(COMPLEX128)

#define FOR_ALL_POPART_TYPES(_)                                                \
  FOR_ALL_FIXED_POINT_TYPES(_)                                                 \
  FOR_ALL_FLOATING_POINT_TYPES(_)                                              \
  _(STRING)                                                                    \
  _(UNDEFINED)

// The types supported by popart.
#define DEFINE_ENUM(value) value,
enum class PopartType { FOR_ALL_POPART_TYPES(DEFINE_ENUM) };
#undef DEFINE_ENUM

#define DEFINE_CASE(value)                                                     \
  case PopartType::value: {                                                    \
    return #value;                                                             \
  }

inline std::string toPopartTypeStr(const PopartType &type) {
  switch (type) {
    FOR_ALL_POPART_TYPES(DEFINE_CASE)
  default:
    ERROR("Unsupported PopartType");
  }
}
#undef DEFINE_CASE

// Returns the dtype int corresponding to the onnx type string
int64_t dtypeIntFromOnnxStr(const char *onnx_type);

// Returns the Onnx datatype as string corresponding the dtype int used in Onnx
// and Popart ops which take an int64_t dtype argument, a.g. "randomnormal"
const char *onnxStrFromDtypeInt(int64_t dtype);

// See popart DataFlow.hpp for a full description of each.
// Must be kept in sync with AnchorMode in python/__init__.py
enum PopartAnchorTypes : std::uint8_t { Final = 0, EveryN, All, Sum, N };

// Must be static so each library gets its own copy,  __attribute__((unused)) is
// to silence the warning if it is unused in any of them.
static PopartAnchorTypes anchorTypeFromString(const std::string &str)
    __attribute__((unused));
static const char *anchorTypeToString(PopartAnchorTypes type)
    __attribute__((unused));

static PopartAnchorTypes anchorTypeFromString(const std::string &str) {
  if (str == "FINAL") {
    return PopartAnchorTypes::Final;
  }
  if (str == "EVERYN") {
    return PopartAnchorTypes::EveryN;
  }
  if (str == "ALL") {
    return PopartAnchorTypes::All;
  }
  if (str == "SUM") {
    return PopartAnchorTypes::Sum;
  }

  ERROR("Internal error: unsupported anchor type :" << str);
}

// Popart only supports a string interface for them so we have to convert back.
static const char *anchorTypeToString(PopartAnchorTypes type) {
  switch (type) {
  case PopartAnchorTypes::Final:
    return "FINAL";
  case PopartAnchorTypes::EveryN:
    return "EVERYN";
  case PopartAnchorTypes::All:
    return "ALL";
  case PopartAnchorTypes::Sum:
    return "Sum";
  default:
    ERROR("UNREACHABLE: Converting anchor type to string");
  }
}

} // namespace poptorch

#endif // POPART_COMPILER_POPART_ENUMS_HPP
