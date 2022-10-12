// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_LOGGING_H
#define INCLUDE_POPTORCH_LOGGING_H

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
#include <string>
#include <utility>

#include "poptorch_logging/LoggingLight.hpp"

/// This is a simple logging system for poptorch based on spdlog. The easiest
/// way to use it is to simply call `logging::<level>()` where <level> is one
/// of trace, debug, info, warn or err. For example:
///
///   #include <core/logging/logging.hpp>
///
///   void foo(int i) {
///     logging::info("foo({}) called", i);
///   }
///
/// logging can be configured by the methods below, or by environment
/// variables, eg
/// POPTORCH_LOG_LEVEL=ERR
/// POPTORCH_LOG_DEST=Mylog.txt
///
/// Formatting is done using the `fmt` library. It supports {}-style and %-style
/// format specification strings. See https://github.com/fmtlib/fmt for details.

namespace poptorch {
namespace logging {

// Log a formatted message. This uses the `fmt` C++ library for formatting.
// See https://github.com/fmtlib/fmt for details. You should probably use
// the MAKE_LOG_TEMPLATE macros instead, e.g.
// logging::debug("The answer is: {}", 42).
template <typename... Args>
void log(Level l, const char *s, const Args &...args) {
  // Avoid formatting if the logging is disabled anyway.
  if (shouldLog(l)) {
    const std::string str = fmt::format(s, args...);
    log(l, str.c_str());
  }
}

// Create a bit of syntactic sugar which allows log statements
// of the form logging::debug("Msg").
#define MAKE_LOG_TEMPLATE(fnName, lvl)                                         \
  template <typename... Args>                                                  \
  inline void fnName(const char *s, const Args &...args) {                     \
    log(Level::lvl, s, std::forward<const Args>(args)...);                     \
  }                                                                            \
                                                                               \
  template <typename... Args>                                                  \
  inline void fnName(std::uint64_t &dedup_count, const char *s,                \
                     const Args &...args) {                                    \
    std::uint64_t rlimit = repeatLimit();                                      \
    if (dedup_count > rlimit) {                                                \
      return;                                                                  \
    }                                                                          \
    if (dedup_count < rlimit) {                                                \
      log(Level::lvl, s, std::forward<const Args>(args)...);                   \
    } else {                                                                   \
      log(Level::lvl, "...repeated messages suppressed...");                   \
    }                                                                          \
    dedup_count++;                                                             \
  }

MAKE_LOG_TEMPLATE(trace, Trace)
MAKE_LOG_TEMPLATE(debug, Debug)
MAKE_LOG_TEMPLATE(info, Info)
MAKE_LOG_TEMPLATE(warn, Warn)
MAKE_LOG_TEMPLATE(err, Err)

#undef MAKE_LOG_TEMPLATE

// Convenience macro to create a log entry prefixed with function name e.g.:
//    void someFunc(int i) {
//      FUNC_LOGGER(info, " with i := {}", i);
//    }
// Then the log entry would be something like:
// 14:30:31.00 [I] void someFunc(int): with i := 42
// NOTE: Because of the limitations of __VA_ARGS__ this log entry must have at
// least one parameter.
#define FUNC_LOGGER(lvl, fmtStr, ...)                                          \
  logging::lvl("{}: " fmtStr, __PRETTY_FUNCTION__, __VA_ARGS__)

#undef FUNC_LOGGER

} // namespace logging
} // namespace poptorch

#endif // INCLUDE_POPTORCH_LOGGING_H
