// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP
#define INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP

#include <torch/csrc/jit/ir/ir.h>

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "popart_compiler/Compiler.hpp"
#include "poptorch/InplaceOps.hpp"

namespace poptorch {

class PoplarExecutable {
public:
  PoplarExecutable() = delete;
  PoplarExecutable(popart_compiler::Compiler &&c,
                   std::vector<popart_compiler::TensorId> &&inputs,
                   std::vector<popart_compiler::TensorId> &&outputs,
                   std::vector<at::ScalarType> &&outputTypes,
                   std::vector<std::string> parameter_names,
                   InplaceGraphInfo &&inplace_info)
      : _compiler(std::move(c)), _popart_inputs(inputs),
        _popart_outputs(outputs), _popart_output_types(outputTypes),
        _parameter_names(std::move(parameter_names)),
        _inplace_info(std::move(inplace_info)) {
    for (size_t i = 0; i < inputs.size(); i++) {
      _converted_inputs.emplace_back();
    }
  }

  void loadEngineAndConnectStreams();
  /*
   * Execute the compiled graph stored in field "compiler" with the given
   * |inTensors| and return to the user the resulting tensors if any.
   */
  std::vector<at::IValue> run(std::vector<at::Tensor> &inTensors);

  void
  updateOptimizers(const std::vector<popart_compiler::Optimizer> &optimizer);

  // Tell popart to copy weights off the IPU and write into host memory.
  void copyWeightsToHost(const std::map<std::string, void *> &buffers);

  // Tell popart to copy weights from host into IPU memory.
  void copyWeightsToDevice(const std::map<std::string, void *> &buffers);

  const std::vector<popart_compiler::OutputTypeShape> &outputTypes() const;

  // Get the IR from popart.
  std::string getPopartIR() const;

  // Get the tensor names that occur in the model graphs.
  std::set<std::string> getTensorNames() const;

  void detachFromDevice();
  void attachToDevice();
  bool isAttachedToDevice() const;

  const popart_compiler::Compiler &getCompiler() const { return _compiler; }
  popart_compiler::Compiler &getCompiler() { return _compiler; }

private:
  popart_compiler::Compiler _compiler;

  std::vector<popart_compiler::TensorId> _popart_inputs;

  // Used for types which need conversion to maintain the ref count
  std::vector<at::Tensor> _converted_inputs;

  std::vector<popart_compiler::TensorId> _popart_outputs;
  std::vector<at::ScalarType> _popart_output_types;
  const std::vector<std::string> _parameter_names;

  const InplaceGraphInfo _inplace_info;
};

} // namespace poptorch

#endif // INCLUDE_POPTORCH_POPLAR_EXECUTABLE_HPP
