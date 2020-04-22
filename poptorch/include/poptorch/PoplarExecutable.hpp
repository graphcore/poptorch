#ifndef INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP
#define INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP

#include <popart_compiler/Compiler.hpp>
#include <torch/csrc/jit/ir/ir.h>

namespace poptorch {

class PoplarExecutable {
public:
  PoplarExecutable() = delete;
  PoplarExecutable(poptorch::Compiler &&c,
                   std::vector<poptorch::TensorId> &&inputs,
                   std::vector<poptorch::TensorId> &&outputs)
      : compiler(std::move(c)), popartInputs(inputs), popartOutputs(outputs) {}

  at::IValue Run(std::vector<at::Tensor> &inTensors);

private:
  poptorch::Compiler compiler;

  std::vector<poptorch::TensorId> popartInputs;

  std::vector<poptorch::TensorId> popartOutputs;
};

} // namespace poptorch

#endif // INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP
