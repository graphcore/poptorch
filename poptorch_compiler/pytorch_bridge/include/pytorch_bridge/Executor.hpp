// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_EXECUTOR_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_EXECUTOR_HPP_
#include <memory>
#include <vector>

namespace poptorch_ir {

namespace detail {
class PoptorchExecutorWrapperImpl;
class PoptorchCompilerImpl;
} // namespace detail

/*
 * A lightweight wrapper around the compiled poplar objects. Allows for the
 * compiler to return an executable back to PyTorch land which can then be
 * maintained independently of the compiler allowing it to freed.
 */
class PoptorchExecutorWrapper {
public:
  ~PoptorchExecutorWrapper();
  void compile(detail::PoptorchCompilerImpl &compiler_impl);
  void execute(const std::vector<void *> &ptrs);

  void weightsToDevice();
  void weightsToHost();

private:
  std::shared_ptr<detail::PoptorchExecutorWrapperImpl> impl;
};

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_EXECUTOR_HPP_
