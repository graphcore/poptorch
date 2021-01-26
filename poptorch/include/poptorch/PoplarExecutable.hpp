// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP
#define INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP

#include <torch/csrc/jit/ir/ir.h>

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "popart_compiler/Compiler.hpp"

namespace poptorch {

class PoplarExecutable {
public:
  PoplarExecutable() = delete;
  PoplarExecutable(poptorch::Compiler &&c,
                   std::vector<poptorch::TensorId> &&inputs,
                   std::vector<poptorch::TensorId> &&outputs,
                   std::vector<at::ScalarType> &&outputTypes,
                   std::vector<std::string> parameter_names)
      : _compiler(std::move(c)), _popart_inputs(inputs),
        _popart_outputs(outputs), _popart_output_types(outputTypes),
        _parameter_names(std::move(parameter_names)) {
    for (size_t i = 0; i < inputs.size(); i++) {
      _converted_inputs.emplace_back();
    }
  }

  /*
   * Execute the compiled graph stored in field "compiler" with the given
   * |inTensors| and return to the user the resulting tensors if any.
   */
  std::vector<at::IValue> run(std::vector<at::Tensor> *inTensors,
                              const std::vector<Optimizer> &optimizer);

  // Tell popart to copy weights off the IPU and write into host memory.
  void copyWeightsToHost(const std::map<std::string, void *> &buffers);

  // Tell popart to copy weights from host into IPU memory.
  void copyWeightsToDevice(const std::map<std::string, void *> &buffers);

  const std::vector<OutputType> &outputTypes() const;

  // Get the IR from popart.
  std::string getPopartIR() const;

  void detachFromDevice();
  void attachToDevice();
  bool isAttachedToDevice() const;

private:
  poptorch::Compiler _compiler;

  std::vector<poptorch::TensorId> _popart_inputs;

  // Used for types which need conversion to maintain the ref count
  std::vector<at::Tensor> _converted_inputs;

  std::vector<poptorch::TensorId> _popart_outputs;
  std::vector<at::ScalarType> _popart_output_types;
  const std::vector<std::string> _parameter_names;
};

} // namespace poptorch

#endif // INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP
