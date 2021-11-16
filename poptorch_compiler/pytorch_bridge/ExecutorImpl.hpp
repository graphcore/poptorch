// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_COMPILER_PYTORCH_BRIDGE_EXECUTOR_IMPL_HPP_
#define POPTORCH_COMPILER_PYTORCH_BRIDGE_EXECUTOR_IMPL_HPP_

#include <string>
#include <utility>
#include <vector>

#include "lower_to_poplar/PoplarExecutor.hpp"
#include "pytorch_bridge/PoptorchCompiler.hpp"

namespace poptorch_ir {
namespace detail {

class PoptorchExecutorWrapperImpl {
public:
  PoptorchExecutorWrapperImpl(std::vector<std::string> &&callbacks,
                              poptorch_ir::PoplarExecutable &&exe)
      : input_callbacks(std::move(callbacks)), executable(std::move(exe)) {}

  ~PoptorchExecutorWrapperImpl();

  // Input and output callbacks to give to poplar.
  std::vector<std::string> input_callbacks;

  // The executable.
  poptorch_ir::PoplarExecutable executable;
};

} // namespace detail

} // namespace poptorch_ir

#endif // POPTORCH_COMPILER_PYTORCH_BRIDGE_EXECUTOR_IMPL_HPP_
