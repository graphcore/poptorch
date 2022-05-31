// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_POPLAR_EXECUTOR_WRAPPER_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_POPLAR_EXECUTOR_WRAPPER_HPP_
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "CompilerTypes.hpp"

namespace poptorch_ir {

namespace detail {
class PoplarExecutorWrapperImpl;
} // namespace detail

class PoplarExecutor;

/*
 * A lightweight wrapper around the compiled poplar objects. Allows for the
 * compiler to return an executor back to PyTorch land which can then be
 * maintained independently of the compiler allowing it to freed.
 */
class PoplarExecutorWrapper {
public:
  ~PoplarExecutorWrapper();
  PoplarExecutorWrapper(
      PoplarExecutor &&executor,
      std::vector<std::pair<std::string, Buffer>> &&input_callbacks);
  void execute(const std::vector<void *> &ptrs);

  void weightsToDevice();
  void weightsToHost();

private:
  std::shared_ptr<detail::PoplarExecutorWrapperImpl> _impl;
};

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_POPLAR_EXECUTOR_WRAPPER_HPP_
