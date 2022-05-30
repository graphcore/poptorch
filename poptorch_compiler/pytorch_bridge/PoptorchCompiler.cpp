// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "pytorch_bridge/PoptorchCompiler.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/Timing.h>

#include <iostream>
#include <utility>

#include "PoptorchCompilerImpl.hpp"
#include "lower_to_poplar/CompilerHelpers.hpp"
#include "pytorch_bridge/PytorchBridgeUtils.hpp"
#include <model_runtime/DeviceManager.hpp>

namespace poptorch_ir {

PoptorchCompiler::PoptorchCompiler() {}
PoptorchCompiler::~PoptorchCompiler() {
  // The timer crashes on destruction when it's enabled. It looks like it's
  // failing to print timing info
  _impl->timing_manager.setEnabled(false);
}

void PoptorchCompiler::init() {
  _impl = std::make_unique<detail::PoptorchCompilerImpl>();
}

TensorId PoptorchCompiler::addInput(const Buffer &ptr,
                                    const std::vector<std::int64_t> &shape,
                                    Type type, const char *name) {
  mlir::RankedTensorType tensor = _impl->getTensor(type, shape);

  // Add the argument to the function args.
  mlir::Value val = _impl->addArgumentToMainGraph(tensor);

  _impl->createOp<poptorch_ir::copy_from_host>(AddToGraph::MAIN_GRAPH, val,
                                               name);

  _impl->value_map.push_back(val);
  _impl->input_callbacks.push_back({name, ptr});

  return _impl->value_map.size() - 1;
}

void PoptorchCompiler::setCurrentPythonCodeLocation(const char *filename,
                                                    std::uint64_t line,
                                                    std::uint64_t col) {
  // If the compiler isn't initialised yet: just ignore the call.
  if (_impl) {
    _impl->setLoc(filename, line, col);
  }
}

TensorId PoptorchCompiler::addParameter(const Buffer &ptr,
                                        const std::vector<std::int64_t> &shape,
                                        Type type, const char *name) {
  mlir::RankedTensorType tensor = _impl->getTensor(type, shape);

  // Add the argument to the function args. Add to the main graph and both
  // weight copy graphs.
  mlir::Value val = _impl->addArgumentToMainGraph(tensor);

  // Write weights to the graph.
  _impl->createOp<poptorch_ir::copy_from_host>(AddToGraph::WRITE_WEIGHTS, val,
                                               "Write-" + std::string(name));

  // Read weights from the graph.
  _impl->createOp<poptorch_ir::copy_to_host>(AddToGraph::READ_WEIGHTS, val,
                                             "Read-" + std::string(name));

  // Add the main graph reference. Both copies are implicitly updating the main
  // graph reference.
  _impl->value_map.push_back(val);

  _impl->weight_callbacks.push_back({name, ptr});
  return _impl->value_map.size() - 1;
}

void PoptorchCompiler::addOutput(TensorId id, void *ptr, const char *name) {
  mlir::Value val = _impl->value_map[id];
  _impl->createOp<poptorch_ir::copy_to_host>(AddToGraph::MAIN_GRAPH, val, name);
  _impl->output_callbacks.push_back({name, ptr});
}

void PoptorchCompiler::startTraceTiming() {
  _impl->timing_manager.setEnabled(true);
  _impl->root_timer = _impl->timing_manager.getRootScope();
  _impl->tracer_timer = _impl->root_timer.nest("PytorchTracingTime");
}

void PoptorchCompiler::endTraceTiming() {
  // Stop the timer
  _impl->tracer_timer = mlir::TimingScope();
}

void PoptorchCompiler::getTimingInfo() { _impl->timing_manager.dumpAsTree(); }

void PoptorchCompiler::dump() { _impl->dump(); }

bool PoptorchCompiler::isView(poptorch_ir::TensorId id) const {
  mlir::Operation *op = _impl->value_map[id].getDefiningOp();

  return op->hasTrait<mlir::OpTrait::ViewOp>();
}

std::vector<std::int64_t> PoptorchCompiler::getSize(TensorId id) const {
  return getRankedTensorType(id).getShape();
}

Type PoptorchCompiler::getType(TensorId id) const {
  return mlirTypeToCompilerType(getRankedTensorType(id).getElementType());
}

mlir::RankedTensorType
PoptorchCompiler::getRankedTensorType(TensorId id) const {
  mlir::Value val = _impl->value_map[id];
  mlir::Type t1 = val.getType();
  return t1.cast<mlir::RankedTensorType>();
}

void PoptorchCompiler::addReturn() {
  // Add returns to each of the graphs.
  _impl->createOp<poptorch_ir::end_graph>(AddToGraph::MAIN_GRAPH);
  _impl->createOp<poptorch_ir::end_graph>(AddToGraph::WRITE_WEIGHTS);
  _impl->createOp<poptorch_ir::end_graph>(AddToGraph::READ_WEIGHTS);
}

bool PoptorchCompiler::allOpsCanBeLoweredToPoplar() const {
  return _impl->all_ops_can_be_lowered;
}

PoplarExecutorWrapper PoptorchCompiler::compileAndLoad() {
  auto device = getDevice();
  auto exe = _impl->compile(device->device().getTarget());
  exe.load(device->device());

  // Connect up the outputs.
  for (auto &pair : _impl->output_callbacks) {
    exe.connectStream(pair.first, pair.second);
  }

  for (auto &pair : _impl->weight_callbacks) {
    exe.connectStream("Write-" + pair.first, pair.second);
    exe.connectStream("Read-" + pair.first, pair.second);
  }

  PoplarExecutorWrapper executor(std::move(exe),
                                 std::move(_impl->input_callbacks));
  _impl->weight_callbacks.clear();
  // input_callbacks was moved to PoplarExecutor in compile()
  return executor;
}

} // namespace poptorch_ir
