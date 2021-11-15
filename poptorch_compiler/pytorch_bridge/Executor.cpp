// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CompilerImpl.hpp"
#include "ExecutorImpl.hpp"

namespace poptorch_ir {

namespace detail {
PoptorchExecutorWrapperImpl::~PoptorchExecutorWrapperImpl() {}
} // namespace detail

PoptorchExecutorWrapper PoptorchCompiler::compile() {
  impl->executable.compile(impl->timing_manager);

  // Connect up the outputs.
  for (auto &pair : impl->output_callbacks) {
    impl->executable.connectStream(pair.first, pair.second);
  }

  for (auto &pair : impl->weight_callbacks) {
    impl->executable.connectStream("Write-" + pair.first, pair.second);
    impl->executable.connectStream("Read-" + pair.first, pair.second);
  }

  PoptorchExecutorWrapper executor;
  executor.compile(*impl);
  return executor;
}

PoptorchExecutorWrapper::~PoptorchExecutorWrapper() {}

void PoptorchExecutorWrapper::compile(
    detail::PoptorchCompilerImpl &compiler_impl) {

  impl = std::make_shared<detail::PoptorchExecutorWrapperImpl>(
      std::move(compiler_impl.input_callbacks),
      std::move(compiler_impl.executable));
}

void PoptorchExecutorWrapper::execute(const std::vector<void *> &ptrs) {

  // Connect up the inputs.
  for (std::size_t i = 0; i < impl->input_callbacks.size(); ++i) {
    impl->executable.connectStream(impl->input_callbacks[i], ptrs[i]);
  }

  // Execute.
  impl->executable.execute();
}

void PoptorchExecutorWrapper::weightsToDevice() {
  impl->executable.weightsToDevice();
}

void PoptorchExecutorWrapper::weightsToHost() {
  impl->executable.weightsToHost();
}

} // namespace poptorch_ir
