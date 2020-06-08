// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPART_COMPILER_ERROR_HPP
#define INCLUDE_POPART_COMPILER_ERROR_HPP

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace poptorch {

/**
 * Exception class for poptorch
 */
class error : public std::runtime_error {
public:
  explicit error(const char *s);
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

void rethrowErrorAsPoplar(const std::exception &e);

} // namespace poptorch

#endif // INCLUDE_POPART_COMPILER_ERROR_HPP
