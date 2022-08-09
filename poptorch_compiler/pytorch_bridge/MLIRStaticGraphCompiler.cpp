// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "MLIRStaticGraphCompiler.hpp"

#include <string>

#include "IMLIRCompiler.hpp"

namespace poptorch_ir {
namespace detail {

MLIRStaticGraphCompiler::MLIRStaticGraphCompiler(
    const poptorch::CompilerOptions &options)
    : IMLIRCompiler(options) {
  _write_weights_graph = createSubGraph("WeightsToDevice");
  _read_weights_graph = createSubGraph("WeightsToHost");
}

// Compile graph by running both PopTorch compiler passes and poplar
// compilation.
poptorch_ir::PoplarExecutor
MLIRStaticGraphCompiler::compile(const PoplarTarget &target) {
  // Start the timer if it has not been started already
  timing_manager.setEnabled(true);
  root_timer.start();

  poptorch_ir::PoplarExecutor exe =
      compileExecutable(_the_module, target, root_timer);

  // End timing
  root_timer.stop();
  timing_manager.setEnabled(false);
  return exe;
}

TensorId MLIRStaticGraphCompiler::addInput(const Buffer &ptr,
                                           const mlir::RankedTensorType &input,
                                           const char *name) {
  // Add the argument to the graph args.
  mlir::Value val = addArgumentToMainGraph(input);

  // Add the argument to the function args.
  createOp<poptorch_ir::copy_from_host>(val, name);
  input_callbacks.push_back({name, ptr});
  return addValue(val);
}

TensorId
MLIRStaticGraphCompiler::addParameter(const Buffer &ptr,
                                      const mlir::RankedTensorType &parameter,
                                      const char *name) {
  // Add the argument to the graph args.
  mlir::Value val = addArgumentToMainGraph(parameter);

  // Write weights to the graph.
  createOp<poptorch_ir::copy_from_host>(_write_weights_graph, val,
                                        "Write-" + std::string(name));

  // Read weights from the graph.
  createOp<poptorch_ir::copy_to_host>(_read_weights_graph, val,
                                      "Read-" + std::string(name));

  weight_callbacks.push_back({name, ptr});
  return addValue(val);
}

void MLIRStaticGraphCompiler::addOutput(void *ptr, TensorId id,
                                        const char *name) {
  mlir::Value output = findValue(id);
  createOp<poptorch_ir::copy_to_host>(output, name);
  output_callbacks.push_back({name, ptr});
}

void MLIRStaticGraphCompiler::addReturn() {
  createOp<poptorch_ir::end_graph>(_write_weights_graph);
  createOp<poptorch_ir::end_graph>(_read_weights_graph);
}

} // namespace detail

} // namespace poptorch_ir
