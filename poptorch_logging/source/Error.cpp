// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poptorch_logging/Error.hpp>
#include <poptorch_logging/Logging.hpp>

namespace logging {

namespace detail {
struct LogContextImpl {
  LogContextImpl() : saved_context(context), cleared(true) {}
  std::string saved_context;
  bool cleared;
  static std::string context;
  static bool trace_enabled;
};
std::string LogContextImpl::context = ""; // NOLINT
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
  _impl->context = _impl->saved_context + " " + new_context;
  _impl->cleared = false;
  if (detail::LogContextImpl::trace_enabled) {
    logging::trace("[{}] Start", _impl->context);
  }
}

void LogContext::clear() {
  if (!_impl->cleared) {
    // Don't restore the saved context if we're handling an exception
    // we might want to recover the context later.
    if (!std::uncaught_exceptions()) {
      if (detail::LogContextImpl::trace_enabled && !_impl->context.empty()) {
        logging::trace("[{}] End", _impl->context);
      }
      // Don't restore the saved context if the context has been cleared.
      if (!_impl->context.empty()) {
        _impl->context = _impl->saved_context;
      }
    }
    _impl->cleared = true;
  }
}

LogContext::~LogContext() { clear(); }

/* static */ const char *LogContext::context() {
  return detail::LogContextImpl::context.c_str();
}
/* static */ void LogContext::resetContext() {
  return detail::LogContextImpl::context.clear();
}
/* static */ bool LogContext::isEmpty() {
  return detail::LogContextImpl::context.empty();
}

} // namespace logging
