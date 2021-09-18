// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poptorch_logging/Error.hpp"

#include <vector>

#include "poptorch_logging/Logging.hpp"

namespace poptorch {
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
  if (level == nullptr) {
    return false;
  }
  return std::string(level) == "TRACE_ALL";
}();

struct ErrorImpl {
  std::string file;
  uint64_t line;
};

} // namespace detail

Error::~Error() = default;

Error::Error(Error &&e)
    : std::runtime_error(e.what()), _impl(std::move(e._impl)) {}

Error::Error(const char *s, const char *file, uint64_t line)
    : std::runtime_error(std::string(s)),
      _impl(std::make_unique<detail::ErrorImpl>()) {
  _impl->file = logging::shortPoptorchFilename(file);
  _impl->line = line;
}

const char *Error::file() const { return _impl->file.c_str(); }

uint64_t Error::line() const { return _impl->line; }

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
    if (std::uncaught_exceptions() == 0) {
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
  if (ctx.empty()) {
    return nullptr;
  }
  for (int64_t idx = ctx.size() - 1; idx >= 0; --idx) {
    ss << "  [" << ctx.size() - idx - 1 << "] " << ctx.at(idx) << std::endl;
  }

  std::string str = ss.str();
  auto ptr = std::unique_ptr<char[]>(new char[str.size() + 1]);
  str.copy(ptr.get(), std::string::npos);
  ptr.get()[str.size()] = '\0';
  return ptr;
}

/* static */ void LogContext::resetContext() { return getContext().clear(); }
/* static */ void LogContext::push(const char *new_context) {
  getContext().push_back(new_context);
}

} // namespace logging
} // namespace poptorch
