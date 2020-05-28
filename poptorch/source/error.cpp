// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poptorch/error.hpp>
#include <poputil/exceptions.hpp>

#include <popart/error.hpp>
#include <poplar/exceptions.hpp>

namespace poptorch {

ErrorSource getErrorSource(const std::exception &e) {
  if (dynamic_cast<const poptorch::internal_error *>(&e)) {
    return ErrorSource::poptorch_internal;
  }
  if (dynamic_cast<const poptorch::error *>(&e)) {
    return ErrorSource::poptorch;
  }
  if (dynamic_cast<const popart::internal_error *>(&e)) {
    return ErrorSource::popart_internal;
  }
  if (dynamic_cast<const popart::memory_allocation_err *>(&e)) {
    return ErrorSource::popart;
  }
  if (dynamic_cast<const popart::error *>(&e)) {
    return ErrorSource::popart;
  }
  if (dynamic_cast<const poplar::poplar_error *>(&e)) {
    return ErrorSource::poplar;
  }
  if (dynamic_cast<const poputil::poplibs_error *>(&e)) {
    return ErrorSource::poplibs;
  }
  return ErrorSource::unknown;
}

} // namespace poptorch
