// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "MLIRStaticGraphCompiler.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <string>

#include "IMLIRCompiler.hpp"
#include "pytorch_bridge/CompilerTypes.hpp"
#include "pytorch_bridge/IpuSession.hpp"
#include "pytorch_bridge/PytorchBridgeUtils.hpp"

namespace poptorch_ir {

namespace {

std::vector<char> toVectorChar(std::string_view str) {
  return std::vector<char>(str.begin(), str.end());
}

} // namespace

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

  // Each block needs a terminator.
  createOpInEpilogue<mlir::func::ReturnOp>(_main_graph);
  createOpInEpilogue<mlir::func::ReturnOp>(_write_weights_graph);
  createOpInEpilogue<mlir::func::ReturnOp>(_read_weights_graph);

  poptorch_ir::PoplarExecutor exe =
      compileExecutable(_the_module, target, root_timer);

  // End timing
  root_timer.stop();
  timing_manager.setEnabled(false);
  return exe;
}

TensorId MLIRStaticGraphCompiler::addInput(const mlir::RankedTensorType &input,
                                           const char *name) {
  // Add the argument to the function args.
  auto val = createOp<poptorch_ir::copy_from_host>(input, name);
  input_callbacks.push_back(poptorch_ir::StreamInfo{
      toVectorChar(name), nullptr, mlirTypeToCompilerType(input)});
  return addValue(val);
}

TensorId MLIRStaticGraphCompiler::addParameter(
    Buffer &ptr, const mlir::RankedTensorType &parameter_type,
    const char *name) {
  auto memref_type = mlir::MemRefType::get(parameter_type.getShape(),
                                           parameter_type.getElementType());
  // Add the argument to the graph args.
  addGlobalState(name, memref_type);

  // Write weights to the graph.
  {
    auto from_host_val =
        createOp<poptorch_ir::copy_from_host>(
            _write_weights_graph, parameter_type, "Write-" + std::string(name))
            .result();
    createOp<poptorch_ir::copy_to_global_state>(_write_weights_graph,
                                                from_host_val, name);
  }

  // Read weights from the graph.
  {
    auto to_host_val = createOp<copy_from_global_state>(_read_weights_graph,
                                                        parameter_type, name)
                           .tensor();
    createOp<poptorch_ir::copy_to_host>(_read_weights_graph, to_host_val,
                                        "Read-" + std::string(name));
  }

  weight_callbacks.push_back(
      poptorch_ir::StreamInfo{toVectorChar(name), ptr.getCpuData(),
                              mlirTypeToCompilerType(parameter_type)});

  // Read global state from the main graph
  auto main_val =
      createOp<copy_from_global_state>(_main_graph, parameter_type, name)
          .tensor();
  createOpInEpilogue<copy_to_global_state>(_main_graph, main_val, name);
  return addValue(main_val);
}

void MLIRStaticGraphCompiler::addOutput(TensorId id, const char *name) {
  // The output location will be set when the program is executed
  const mlir::Value output = findValue(id);

  // Don't bother adding an output if there is nothing to output
  if (output.getType().cast<mlir::RankedTensorType>().getNumElements() != 0) {
    createOp<poptorch_ir::copy_to_host>(output, name);
  }

  output_callbacks.push_back(poptorch_ir::StreamInfo{
      toVectorChar(name), nullptr,
      mlirTypeToCompilerType(output.getType().cast<mlir::RankedTensorType>())});
}

} // namespace detail

} // namespace poptorch_ir
