// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "pytorch_bridge/PoplarExecutorWrapper.hpp"

#include <algorithm>
#include <iterator>

#include "lower_to_poplar/PoplarExecutor.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch_ir {

namespace detail {
class PoplarExecutorWrapperImpl {
public:
  PoplarExecutorWrapperImpl(std::vector<StreamInfo> &&input_callbacks_local,
                            std::vector<StreamInfo> &&output_callbacks_local,
                            poptorch_ir::PoplarExecutor &&exe)
      : input_callbacks(std::move(input_callbacks_local)),
        output_callbacks(std::move(output_callbacks_local)),
        executor(std::move(exe)) {}

  ~PoplarExecutorWrapperImpl() {}

  // Input and output callbacks to give to poplar.
  std::vector<StreamInfo> input_callbacks;
  std::vector<StreamInfo> output_callbacks;

  // The executor.
  poptorch_ir::PoplarExecutor executor;
};
} // namespace detail

PoplarExecutorWrapper::~PoplarExecutorWrapper() {}

PoplarExecutorWrapper::PoplarExecutorWrapper(
    PoplarExecutor &&executor, std::vector<StreamInfo> &&input_callbacks,
    std::vector<StreamInfo> &&output_callbacks)
    : _impl(std::make_shared<detail::PoplarExecutorWrapperImpl>(
          std::move(input_callbacks), std::move(output_callbacks),
          std::move(executor))) {}

void PoplarExecutorWrapper::execute(const std::vector<void *> &input_ptrs,
                                    const std::vector<void *> &output_ptrs) {
  // Connect up the inputs.
  ERROR_ON(input_ptrs.size() != _impl->input_callbacks.size());
  for (std::size_t i = 0; i < _impl->input_callbacks.size(); ++i) {
    // Did the user provide a new pointer for this input?
    if (input_ptrs[i] != nullptr) {
      // Release the source previously used and switch to the user provided
      // pointer.
      _impl->input_callbacks[i].buff.reset();
      _impl->executor.connectStream(_impl->input_callbacks[i].nameStringView(),
                                    input_ptrs[i]);
    } else {
      ERROR_ON(!_impl->input_callbacks[i].buff);
      _impl->executor.connectStream(_impl->input_callbacks[i].nameStringView(),
                                    _impl->input_callbacks[i].buff->data());
    }
  }

  // Connect up the outputs
  ERROR_ON(output_ptrs.size() != _impl->output_callbacks.size());
  for (std::size_t i = 0; i < _impl->output_callbacks.size(); ++i) {
    // Note that tensors with no elements are still added as outputs but they
    // don't appear in the graph.
    if (_impl->output_callbacks[i].type.getNumElements() != 0) {
      if (output_ptrs[i] != nullptr) {
        // Release the source previously used and switch to the user provided
        // pointer.
        _impl->output_callbacks[i].buff.reset();
        _impl->executor.connectStream(
            _impl->output_callbacks[i].nameStringView(), output_ptrs[i]);
      } else {
        ERROR("Missing output " << i);
      }
    }
  }

  // Execute.
  _impl->executor.execute();
}

std::vector<TensorType> PoplarExecutorWrapper::outputTypes() const {
  std::vector<TensorType> output_types;
  std::transform(_impl->output_callbacks.begin(), _impl->output_callbacks.end(),
                 std::back_inserter(output_types),
                 [](const auto &elt) { return elt.type; });
  return output_types;
}

void PoplarExecutorWrapper::weightsToDevice() {
  _impl->executor.weightsToDevice();
}

void PoplarExecutorWrapper::weightsToHost() { _impl->executor.weightsToHost(); }

} // namespace poptorch_ir
