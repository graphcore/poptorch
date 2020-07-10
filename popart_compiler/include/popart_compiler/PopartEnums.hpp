// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_COMPILER_POPART_ENUMS_HPP
#define POPART_COMPILER_POPART_ENUMS_HPP
#include <string>

#include <poptorch_logging/Error.hpp>

namespace poptorch {

/*
 * We maintain an ABI boundary inbetween PopART and Torch JIT. This avoids the
 * issue of torch being compiled with different CXX_ABI versions. However it
 * means we must replicate PopART enums here so they can be shared by both.
 */

// The training optimizer algorithm used.
enum OptimizerType : std::uint8_t { NONE, SGD };

// The types supported by popart.
enum PopartTypes {
  // fixed point types
  UINT8 = 0,
  INT8,
  UINT16,
  INT16,
  INT32,
  INT64,
  UINT32,
  UINT64,
  BOOL,
  // floating point types
  FLOAT,
  FLOAT16,
  BFLOAT16,
  DOUBLE,
  COMPLEX64,
  COMPLEX128,
  // other types
  STRING,
  UNDEFINED,
};

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
