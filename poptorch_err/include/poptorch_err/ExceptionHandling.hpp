// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once
#include <exception>
#include <functional>
#include <string>
#include <vector>

#include "poptorch_err/ExceptionInfo.hpp"

namespace poptorch {
/*
 * The function convertToPoptorchExceptionOrRethrow() processes all the
 * exception types we're interested in, extracts detail, and marshals them as
 * instances of PoptorchErrorInfo. The exceptions we're not interested in are
 * re-thrown as-is.
 */
struct PoptorchErrorInfo {
public:
  ErrorCategory category;
  std::string filename;
  uint64_t line;
  std::string type;
  std::string recovery_action;
  std::string message;
  std::string long_message;
  std::vector<std::string> stack;
  std::string location;
};

PoptorchErrorInfo
convertToPoptorchExceptionOrRethrow(const std::exception_ptr &e, bool catch_all,
                                    const std::string &catch_file,
                                    uint64_t catch_line);

} // namespace poptorch

/*
 * This template wraps a function in our try..catch block. It's done this way
 * so it's less likely that someone will add an entry point without wrapping
 * it in a try..catch block - the path of least resistance is to copy-paste
 * the pybind11 def() line which will include the PTC() macro.
 * This doesn't work for class member functions wrapped by pybind11, which have
 * to be manually wrapped in a try-catch block.
 */
template <void (*g)(const poptorch::PoptorchErrorInfo &), bool catch_all,
          class F, F f>
struct PoptorchCatchWrapperImpl;
template <void (*g)(const poptorch::PoptorchErrorInfo &), bool catch_all,
          class R, class... Args, R (*f)(Args...)>
struct PoptorchCatchWrapperImpl<g, catch_all, R (*)(Args...), f> {
  static R wrap(Args... args) {
    try {
      return f(args...);
    } catch (...) {
      // TODO(T71675): find a way to pass catch_file / catch_line
      g(poptorch::convertToPoptorchExceptionOrRethrow(std::current_exception(),
                                                      catch_all, "unknown", 0));
    }
  }
};
