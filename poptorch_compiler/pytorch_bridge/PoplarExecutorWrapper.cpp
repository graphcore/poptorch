// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "pytorch_bridge/PoplarExecutorWrapper.hpp"

#include "lower_to_poplar/PoplarExecutor.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch_ir {

namespace detail {
class PoplarExecutorWrapperImpl {
public:
  PoplarExecutorWrapperImpl(
      std::vector<std::pair<std::string, Buffer>> &&callbacks,
      poptorch_ir::PoplarExecutor &&exe)
      : input_callbacks(std::move(callbacks)), executor(std::move(exe)) {}

  ~PoplarExecutorWrapperImpl() {}

  // Input and output callbacks to give to poplar.
  std::vector<std::pair<std::string, Buffer>> input_callbacks;

  // The executor.
  poptorch_ir::PoplarExecutor executor;
};
} // namespace detail

PoplarExecutorWrapper::~PoplarExecutorWrapper() {}

PoplarExecutorWrapper::PoplarExecutorWrapper(
    PoplarExecutor &&executor,
    std::vector<std::pair<std::string, Buffer>> &&input_callbacks)
    : _impl(std::make_shared<detail::PoplarExecutorWrapperImpl>(
          std::move(input_callbacks), std::move(executor))) {}

void PoplarExecutorWrapper::execute(const std::vector<void *> &ptrs) {
  // Connect up the inputs.
  for (std::size_t i = 0; i < _impl->input_callbacks.size(); ++i) {
    // Did the user provide a new pointer for this input?
    if (ptrs[i] != nullptr) {
      // Release the source previously used and switch to the user provided
      // pointer.
      _impl->input_callbacks[i].second.reset();
      _impl->executor.connectStream(_impl->input_callbacks[i].first, ptrs[i]);
    } else {
      ERROR_ON(!_impl->input_callbacks[i].second);
      _impl->executor.connectStream(_impl->input_callbacks[i].first,
                                    _impl->input_callbacks[i].second->data());
    }
  }

  // Execute.
  _impl->executor.execute();
}

void PoplarExecutorWrapper::weightsToDevice() {
  _impl->executor.weightsToDevice();
}

void PoplarExecutorWrapper::weightsToHost() { _impl->executor.weightsToHost(); }

} // namespace poptorch_ir
