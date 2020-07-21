// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP
#define INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP

#include <torch/csrc/jit/ir/ir.h>

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
                   std::vector<at::ScalarType> &&outputTypes)
      : _compiler(std::move(c)), _popartInputs(inputs), _popartOutputs(outputs),
        _popartOutputTypes(outputTypes) {
    for (size_t i = 0; i < inputs.size(); i++) {
      _convertedInputs.emplace_back();
    }
  }

  /*
   * Execute the compiled graph stored in field "compiler" with the given
   * |inTensors| and return to the user the resulting tensors if any.
   */
  std::vector<at::IValue> run(std::vector<at::Tensor> *inTensors,
                              const Optimizer &optimizer);

  // Tell popart to copy weights off the IPU and write into host memory.
  void copyWeightsToHost();

  // Tell popart to copy weights from host into IPU memory.
  void copyWeightsToDevice();

  const std::vector<OutputType> &outputTypes() const;

private:
  poptorch::Compiler _compiler;

  std::vector<poptorch::TensorId> _popartInputs;

  // Used for types which need conversion to maintain the ref count
  std::vector<at::Tensor> _convertedInputs;

  std::vector<poptorch::TensorId> _popartOutputs;
  std::vector<at::ScalarType> _popartOutputTypes;
};

} // namespace poptorch

#endif // INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP
