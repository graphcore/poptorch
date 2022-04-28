// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include <cstdint>

namespace poptorch {

enum class ErrorCategory { RuntimeRecoverable, RuntimeUnrecoverable, Other };

/*
 * A subclass of this class is used to pass exception information across the ABI
 * boundary between popart_compiler and the pybind11 interface. It has to use
 * POD data types to cross the boundary successfully. We then unpack it into a
 * PoptorchError on the pybind11 side and rethrow it.
 */
class ExceptionInfo {
public:
  virtual ~ExceptionInfo();
  const virtual char *what() const = 0;
  const virtual char *type() const = 0;
  virtual int64_t stackDepth() const = 0;
  const virtual char *stack(int64_t level) const = 0;
  const virtual char *filename() const = 0;
  virtual uint64_t line() const = 0;
  const virtual char *recoveryAction() const = 0;
  virtual ErrorCategory category() const = 0;
};

} // namespace poptorch
