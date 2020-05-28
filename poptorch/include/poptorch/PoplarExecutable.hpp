// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP
#define INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP

#include <utility>
#include <vector>

#include <popart_compiler/Compiler.hpp>
#include <torch/csrc/jit/ir/ir.h>

namespace poptorch {

class PoplarExecutable {
public:
  PoplarExecutable() = delete;
  PoplarExecutable(poptorch::Compiler &&c,
                   std::vector<poptorch::TensorId> &&inputs,
                   std::vector<poptorch::TensorId> &&outputs, bool p)
      : compiler(std::move(c)), popartInputs(inputs), popartOutputs(outputs),
        profile(p) {}

  std::vector<at::IValue> Run(std::vector<at::Tensor> &inTensors);

private:
  poptorch::Compiler compiler;

  std::vector<poptorch::TensorId> popartInputs;

  std::vector<poptorch::TensorId> popartOutputs;

  bool profile;
};

} // namespace poptorch

#endif // INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP
