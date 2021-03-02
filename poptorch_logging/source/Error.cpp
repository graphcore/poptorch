// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace logging {

const char *shortPoptorchFilename(const char *filename) {
  auto pos = std::string(filename).rfind("/poptorch/");
  if (pos == std::string::npos) {
    return filename;
  }
  return filename + pos + 1; // NOLINT
}

std::string &getContext() {
  static std::string log_context;
  return log_context;
}

namespace detail {
struct LogContextImpl {
  LogContextImpl() : saved_context(getContext()), cleared(true) {}
  std::string saved_context;
  bool cleared;
  static bool trace_enabled;
};

bool LogContextImpl::trace_enabled = []() {
  auto level = std::getenv("POPTORCH_LOG_LEVEL");
  if (!level) {
    return false;
  }
  return std::string(level) == "TRACE_ALL";
}();
} // namespace detail

Error::Error(const char *s) : std::runtime_error(s) { logging::err(what()); }

LogContext::LogContext() : _impl(std::make_unique<detail::LogContextImpl>()) {}

LogContext::LogContext(const char *context) : LogContext() {
  updateContext(context);
}

void LogContext::updateContext(const std::string &new_context) {
  clear();
  getContext() = _impl->saved_context + " " + new_context;
  _impl->cleared = false;
  if (detail::LogContextImpl::trace_enabled) {
    logging::trace("[{}] Start", getContext());
  }
}

void LogContext::clear() {
  if (!_impl->cleared) {
    // Don't restore the saved context if we're handling an exception
    // we might want to recover the context later.
    if (!std::uncaught_exceptions()) {
      if (detail::LogContextImpl::trace_enabled && !getContext().empty()) {
        logging::trace("[{}] End", getContext());
      }
      // Don't restore the saved context if the context has been cleared.
      if (!getContext().empty()) {
        getContext() = _impl->saved_context;
      }
    }
    _impl->cleared = true;
  }
}

LogContext::~LogContext() { clear(); }

/* static */ const char *LogContext::context() { return getContext().c_str(); }
/* static */ void LogContext::resetContext() { return getContext().clear(); }
/* static */ bool LogContext::isEmpty() { return getContext().empty(); }

} // namespace logging
