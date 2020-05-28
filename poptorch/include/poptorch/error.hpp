// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_ERROR_HPP
#define INCLUDE_POPTORCH_ERROR_HPP

#include <memory>
#include <shared/Logging.hpp>
#include <spdlog/fmt/fmt.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace poptorch {

/**
 * Exception class for poptorch
 */
class error : public std::runtime_error {

private:
  // A type to ensure that the variadic constructors do not get called
  struct _empty {};

  // Constructors that do not throw exception, used in the case that the
  // fmt::format function throws an exception
  explicit error(const _empty &, const char *s) : std::runtime_error(s) {
    logging::err(what());
  }

  explicit error(const _empty &, const std::string &s)
      : std::runtime_error(s) {
    logging::err(what());
  }

public:
  /// Variadic constructor for error which allows the user to use a fmt string
  /// for the message. As the fmt::format function can throw an exception
  /// itself and it is used in the initilization list we have to use the
  /// unusal C++ syntax to catch that exception and convert it to a poptorch
  /// exception
  ///
  /// throw error("This is an error reason {}", 42);

  template <typename... Args>
  explicit error(const char *s, const Args &... args) try : std
    ::runtime_error(fmt::format(s, args...)) { logging::err(what()); }
  catch (const fmt::FormatError &e) {
    std::string reason = std::string("Poptorch exception format error ") +
                         std::string(e.what());
    error _e(_empty(), reason);
    throw _e;
  }

  template <typename... Args>
  explicit error(const std::string &s, const Args &... args) try : std
    ::runtime_error(fmt::format(s, args...)) { logging::err(what()); }
  catch (const fmt::FormatError &e) {
    std::string reason = std::string("Poptorch exception format error:") +
                         std::string(e.what());
    throw error(_empty(), reason);
  }
};

/**
 * Exception class specific to internal errors
 * This should be used as an assert; for states where the user should not have
 * been able to create.
 */
class internal_error : public error {
public:
  using error::error;
};

enum class ErrorSource {
  poptorch = 0,
  poptorch_internal,
  popart,
  popart_internal,
  poplar,
  poplibs,
  unknown,
};

ErrorSource getErrorSource(const std::exception &e);

} // namespace poptorch

#endif // INCLUDE_POPTORCH_ERROR_HPP
