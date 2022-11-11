// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_LOGGING_ERROR_HPP
#define INCLUDE_POPTORCH_LOGGING_ERROR_HPP

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace poptorch {
namespace logging {

namespace detail {
struct LogContextImpl;
struct ErrorImpl;
} // namespace detail

// Remove everything before the last occurrence of "/poptorch/" in a string
// For example given an absolute path like:
// /a/b/c/poptorch/d/e/f.cpp -> poptorch/d/e/f.cpp
const char *shortPoptorchFilename(const char *filename);

#define UNUSED(var) (void)(var)

#define ERROR(msg)                                                             \
  do {                                                                         \
    std::stringstream __error_msg;                                             \
    __error_msg << msg; /* NOLINT */                                           \
    throw ::poptorch::logging::InternalError(__error_msg.str().c_str(),        \
                                             __FILE__, __LINE__);              \
  } while (0)

#define ERROR_ON_MSG(condition, msg)                                           \
  do {                                                                         \
    if (__builtin_expect(static_cast<bool>(condition), 0)) {                   \
      ERROR(msg);                                                              \
    }                                                                          \
  } while (0)

#define ERROR_ON(condition) ERROR_ON_MSG(condition, #condition)

/**
 * Exception class for poptorch.
 *
 * The what() method returns both the error message and the
 * stacktrace.
 * To have the error without the stacktrace use message().
 */
class Error : public std::runtime_error {
public:
  explicit Error(const char *s, const char *file, uint64_t line);
  Error(Error &&e);
  const char *file() const;
  uint64_t line() const;
  // The error message without the stacktrace
  const char *message() const;
  ~Error() override;

private:
  std::unique_ptr<detail::ErrorImpl> _impl;
};

/**
 * Exception class specific to internal errors
 * This should be used as an assert; for states where the user should not have
 * been able to create.
 */
class InternalError : public Error {
public:
  using Error::Error;
};

/* Context stack used to attach extra information to exceptions when they're
 * raised. All contexts changes can be printed by enabling the info mode.
 */
class LogContext {
public:
  // Current context stack as a string
  static std::unique_ptr<char[]> context();
  static void resetContext();
  static void push(const char *);

  LogContext();
  // Push the context at the top of the context stack.
  explicit LogContext(const std::string &context)
      : LogContext(context.c_str()) {}
  explicit LogContext(const char *context);

  // Replace the top of the context stack with new_context.
  void updateContext(const std::string &new_context);

  // Pop the top of the context stack.
  void clear();
  // Implicitly pop the top of the context stack if clear() hasn't been
  // explicitly called.
  ~LogContext();

private:
  std::unique_ptr<detail::LogContextImpl> _impl;
};

} // namespace logging
} // namespace poptorch

#endif // INCLUDE_POPTORCH_LOGGING_ERROR_HPP
