// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart_compiler/Error.hpp>
#include <spdlog/fmt/fmt.h>

#include <popart/error.hpp>
#include <poplar/exceptions.hpp>
#include <poptorch_logging/Logging.hpp>
#include <poputil/exceptions.hpp>

namespace poptorch {

enum class ErrorSource {
  poptorch = 0,
  poptorch_internal,
  popart,
  popart_internal,
  poplar,
  poplibs,
  unknown,
};

void rethrowErrorAsPoplar(const std::exception &e) {
  if (auto *err = dynamic_cast<const poptorch::internal_error *>(&e)) {
    throw std::runtime_error(err->what());
  }
  if (auto *err = dynamic_cast<const poptorch::error *>(&e)) {
    throw std::runtime_error(err->what());
  }
  if (auto *err = dynamic_cast<const popart::internal_error *>(&e)) {
    throw std::runtime_error(err->what());
  }
  if (auto *err = dynamic_cast<const popart::memory_allocation_err *>(&e)) {
    throw std::runtime_error(err->what());
  }
  if (auto *err = dynamic_cast<const popart::error *>(&e)) {
    throw std::runtime_error(err->what());
  }
  if (auto *err = dynamic_cast<const poplar::poplar_error *>(&e)) {
    throw std::runtime_error(err->what());
  }
  if (auto *err = dynamic_cast<const poputil::poplibs_error *>(&e)) {
    throw std::runtime_error(err->what());
  }
}

error::error(const char *s) : std::runtime_error(s) { logging::err(what()); }

} // namespace poptorch
