// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "CompilerImpl.hpp"
#include "ExecutorImpl.hpp"

#include "poptorch_logging/Error.hpp"

namespace poptorch_ir {

namespace detail {
PoptorchExecutorWrapperImpl::~PoptorchExecutorWrapperImpl() {}
} // namespace detail

PoptorchExecutorWrapper PoptorchCompiler::compile() {
  _impl->compile();

  // Connect up the outputs.
  for (auto &pair : _impl->output_callbacks) {
    _impl->connectStream(pair.first, pair.second);
  }

  for (auto &pair : _impl->weight_callbacks) {
    _impl->connectStream("Write-" + pair.first, pair.second);
    _impl->connectStream("Read-" + pair.first, pair.second);
  }

  PoptorchExecutorWrapper executor;
  executor.compile(*_impl);
  _impl->weight_callbacks.clear();
  // input_callbacks was moved to PoplarExecutor in compile()
  return executor;
}

PoptorchExecutorWrapper::~PoptorchExecutorWrapper() {}

void PoptorchExecutorWrapper::compile(
    detail::PoptorchCompilerImpl &compiler_impl) {

  _impl = std::make_shared<detail::PoptorchExecutorWrapperImpl>(
      std::move(compiler_impl.input_callbacks), compiler_impl.getExecutable());
}

void PoptorchExecutorWrapper::execute(const std::vector<void *> &ptrs) {
  // Connect up the inputs.
  for (std::size_t i = 0; i < _impl->input_callbacks.size(); ++i) {
    // Did the user provide a new pointer for this input?
    if (ptrs[i] != nullptr) {
      // Release the source previously used and switch to the user provided
      // pointer.
      _impl->input_callbacks[i].second.reset();
      _impl->executable.connectStream(_impl->input_callbacks[i].first, ptrs[i]);
    } else {
      ERROR_ON(!_impl->input_callbacks[i].second);
      _impl->executable.connectStream(_impl->input_callbacks[i].first,
                                      _impl->input_callbacks[i].second->data());
    }
  }

  // Execute.
  _impl->executable.execute();
}

void PoptorchExecutorWrapper::weightsToDevice() {
  _impl->executable.weightsToDevice();
}

void PoptorchExecutorWrapper::weightsToHost() {
  _impl->executable.weightsToHost();
}

} // namespace poptorch_ir
