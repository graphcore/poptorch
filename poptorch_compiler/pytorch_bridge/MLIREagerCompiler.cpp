// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "MLIREagerCompiler.hpp"

#include <deque>
#include <utility>

#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {
namespace detail {

MLIREagerCompiler::MLIREagerCompiler(PoplarDevice &device,
                                     const poptorch::CompilerOptions &options)
    : IMLIRCompiler(options), _executor(device) {}

TensorId MLIREagerCompiler::addValue(const mlir::Value &value) {
  ERROR_ON(!value);
  _tensor_map.push_back(value.getType().cast<mlir::RankedTensorType>());
  return IMLIRCompiler::addValue(value);
}

mlir::Value MLIREagerCompiler::findValue(TensorId tensor) {
  mlir::Value value = IMLIRCompiler::findValue(tensor);
  // This tensor comes from a previous graph, we need
  // to add it as an input to the new graph.
  if (!value) {
    value = addArgumentToMainGraph(_tensor_map.at(tensor));
    updateTensor(tensor, value);
  }
  return value;
}

void MLIREagerCompiler::compileRunAndReset() {
  root_timer.start();

  auto mappings = getValueMappings();

  std::deque<mlir::Value> outputs;
  // Find all the outputs of all the ops as graph outputs
  _main_graph.graph.walk([&](mlir::Operation *op) {
    if (!op->hasTrait<PoplarImplInterface::Trait>()) {
      return;
    }
    for (auto result : op->getResults()) {
      outputs.push_back(result);
    }
  });

  // Add them all as graph outputs
  for (auto &output : outputs) {
    auto it = mappings.find(output);
    if (it == mappings.end()) {
      poptorch::logging::trace(
          "No tensor ID mapping for {}: not marking as output",
          mlirToStr(output));
    } else {
      createOp<poptorch_ir::output_tensor>(output, it->second);
    }
  }

  _executor.compileAndRun(_the_module, root_timer, std::move(mappings));
  resetMainGraph();

  root_timer.stop();
  timing_manager.setEnabled(false);
}

TensorId MLIREagerCompiler::addInput(const Buffer &ptr,
                                     const mlir::RankedTensorType &input,
                                     const char *name) {
  UNUSED(name);
  TensorId id = IMLIRCompiler::addValue(mlir::Value());
  _tensor_map.push_back(input);
  _executor.addInput(ptr, input, id);

  return id;
}

TensorId
MLIREagerCompiler::addParameter(const Buffer &ptr,
                                const mlir::RankedTensorType &parameter,
                                const char *name) {
  UNUSED(name);
  TensorId id = IMLIRCompiler::addValue(mlir::Value());
  _tensor_map.push_back(parameter);
  _executor.addInput(ptr, parameter, id);
  return id;
}

void MLIREagerCompiler::addOutput(void *ptr, TensorId id, const char *name) {
  UNUSED(name);
  compileRunAndReset();
  _executor.readOutput(id, ptr);
}

void MLIREagerCompiler::onOpAdded() {
  if (shouldRunAllOpsSynchronously()) {
    compileRunAndReset();
  }
}

void MLIREagerCompiler::addReturn() {
  ERROR("Only static graphs have a return");
}

bool MLIREagerCompiler::shouldRunAllOpsSynchronously() const {
  return !_compiler_options.eager.use_lazy_tensor;
}
} // namespace detail

} // namespace poptorch_ir
