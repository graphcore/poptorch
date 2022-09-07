// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_LOGGING_LIGHT_H
#define INCLUDE_POPTORCH_LOGGING_LIGHT_H

#include <string>
#include <utility>

// This header is a lighter version of poptorch_logging which doesn't require
// spdlog and therefore doesn't support formatting.
//
// For the full version of the logging API use
// poptorch_logging/Logging.hpp instead.
namespace poptorch {
namespace logging {

enum class Level {
  Trace = 0,
  Debug = 1,
  Info = 2,
  Warn = 3,
  Err = 4,
  // level 5 is "critical" in spdlog, which we don't use so isn't exposed here.
  Off = 6,
};

// Set the current log level to one of the above levels. The default
// log level is set by the POPTORCH_LOG_LEVEL environment variable
// and is off by default.
void setLogLevel(Level l);

// Return true if the passed log level is currently enabled.
bool shouldLog(Level l);

// Return true if the Popart IR should be dumped.
bool outputPopartIR();

// Flush the log. By default it is only flushed when the underlying libc
// decides to.
void flush();

// Log a message. You should probably use the MAKE_LOG_TEMPLATE macros
// instead, e.g. logging::debug("A debug message").
void log(Level l, const char *msg);

} // namespace logging
} // namespace poptorch

#endif // INCLUDE_POPTORCH_LOGGING_LIGHT_H
