// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once
#include <torch/csrc/Exceptions.h>

#include <exception>
#include <functional>
#include <string>
#include <vector>

#include "poptorch_err/ExceptionInfo.hpp"

namespace poptorch {

/*
 * This structure enables poptorch.Error objects to be thrown python-side from
 * both our pybind11 interface and torch's own. Our pybind11 exception handler
 * catches this class specifically, whilst torch's catches any PyTorchError
 * subclass and uses it to deduce the python type using the overridden
 * python_type() method.
 * The function rethrowPoptorchException() processes all the exception types
 * we're interested in, extracts detail, and marshals them as instances of
 * this class. We put try..catch wrappers round every pybind11 entry point using
 * the macro CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION and pass them to
 * rethrowPoptorchException().
 */
struct PoptorchError : public torch::PyTorchError {
public:
  explicit PoptorchError(const std::string &msg_);
  PyObject *python_type() override;
  void setErrorIndicator() const;

private:
  PyObject *setupPyError(bool set_indicator) const;

public:
  ErrorCategory category;
  std::string filename;
  uint64_t line;
  std::string type;
  std::string recovery_action;
  std::string message;
  std::vector<std::string> stack;
  std::string location;
};

void initialiseExceptionHandling(pybind11::handle m);

void rethrowPoptorchException(const std::exception_ptr &e,
                              const std::string &catch_file,
                              uint64_t catch_line);

} // namespace poptorch

#define CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION                                \
  catch (...) {                                                                \
    poptorch::rethrowPoptorchException(std::current_exception(), __FILE__,     \
                                       __LINE__);                              \
  }

/*
 * This template wraps a function in our try..catch block. It's done this way
 * so it's less likely that someone will add an entry point without wrapping
 * it in a try..catch block - the path of least resistance is to copy-paste
 * the pybind11 def() line which will include the PTC() macro.
 * This doesn't work for class member functions wrapped by pybind11, which have
 * to be manually wrapped in a try-catch block.
 */
template <class F, F f> struct PoptorchCatchWrapperImpl;
template <class R, class... Args, R (*f)(Args...)>
struct PoptorchCatchWrapperImpl<R (*)(Args...), f> {
  static R wrap(Args... args) {
    try {
      return f(args...);
    }
    CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION
  }
};

#define PTC(f) PoptorchCatchWrapperImpl<decltype(&(f)), f>::wrap
#define PTC_BOXED(f) torch::CppFunction::makeFromBoxedFunction<PTC(f)>()
