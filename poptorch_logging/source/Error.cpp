// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poptorch_logging/Error.hpp"

#include <vector>

#include "poptorch_logging/Logging.hpp"

namespace logging {

namespace {
using Context = std::vector<std::string>;

Context &getContext() {
  static Context log_context{};
  return log_context;
}

std::string singleLineContext() {
  std::stringstream ss;
  std::string sep{};
  for (const auto &lvl : getContext()) {
    ss << sep << lvl;
    sep = " -> ";
  }
  return ss.str();
}

} // namespace

const char *shortPoptorchFilename(const char *filename) {
  auto pos = std::string(filename).rfind("/poptorch/");
  if (pos == std::string::npos) {
    return filename;
  }
  return filename + pos + 1; // NOLINT
}

namespace detail {
struct LogContextImpl {
  LogContextImpl() : cleared(true) {}
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
  getContext().push_back(new_context);
  _impl->cleared = false;
  if (detail::LogContextImpl::trace_enabled) {
    logging::trace("[{}] Start", singleLineContext());
  }
}

void LogContext::clear() {
  if (!_impl->cleared) {
    // Don't restore the saved context if we're handling an exception
    // we might want to recover the context later.
    if (!std::uncaught_exceptions()) {
      if (detail::LogContextImpl::trace_enabled && !getContext().empty()) {
        logging::trace("[{}] End", singleLineContext());
      }
      // Don't restore the saved context if the context has been cleared.
      if (!getContext().empty()) {
        getContext().pop_back();
      }
    }
    _impl->cleared = true;
  }
}

LogContext::~LogContext() { clear(); }

/* static */ std::unique_ptr<char[]> LogContext::context() {
  std::stringstream ss;
  auto &ctx = getContext();
  for (int64_t idx = ctx.size() - 1; idx >= 0; --idx) {
    ss << "  [" << ctx.size() - idx - 1 << "] " << ctx.at(idx) << std::endl;
  }

  std::string str = ss.str();
  if (str.empty()) {
    return nullptr;
  }

  auto ptr = std::unique_ptr<char[]>(new char[str.size() + 1]);
  str.copy(ptr.get(), std::string::npos);
  ptr.get()[str.size()] = '\0';
  return ptr;
}

/* static */ void LogContext::resetContext() { return getContext().clear(); }
/* static */ bool LogContext::isEmpty() { return getContext().empty(); }
/* static */ void LogContext::push(const char *new_context) {
  getContext().push_back(new_context);
}

} // namespace logging
