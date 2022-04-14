// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "pytorch_bridge/PoptorchCompiler.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Operation.h>

#include <iostream>
#include <utility>

#include "CompilerImpl.hpp"
#include "ExecutorImpl.hpp"
#include "pytorch_bridge/PytorchBridgeUtils.hpp"

namespace poptorch_ir {

PoptorchCompiler::PoptorchCompiler() {}
PoptorchCompiler::~PoptorchCompiler() {}

void PoptorchCompiler::init() {
  _impl = std::make_unique<detail::PoptorchCompilerImpl>();
}

TensorId PoptorchCompiler::addInput(void * /*unused*/,
                                    const std::vector<std::int64_t> &shape,
                                    Type type, const char *name) {
  mlir::RankedTensorType tensor = _impl->getTensor(type, shape);

  // Add the argument to the function args.
  mlir::Value val = _impl->addArgumentToMainGraph(tensor);

  _impl->createOp<poptorch_ir::copy_from_host>(AddToGraph::MAIN_GRAPH, val,
                                               name);

  _impl->value_map.push_back(val);
  _impl->input_callbacks.push_back(name);

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

TensorId PoptorchCompiler::addParameter(void *ptr,
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

TensorId PoptorchCompiler::addBuffer(void *ptr,
                                     const std::vector<std::int64_t> &shape,
                                     Type type, const char *name) {
  return addParameter(ptr, shape, type, name);
}

void PoptorchCompiler::addOutput(TensorId id, void *ptr, const char *name) {
  mlir::Value val = _impl->value_map[id];
  _impl->createOp<poptorch_ir::copy_to_host>(AddToGraph::MAIN_GRAPH, val, name);
  _impl->output_callbacks.push_back({name, ptr});
}

void PoptorchCompiler::startTraceTiming() {
  _impl->timing_manager.start("PytorchTracingTime");

  // TODO(T49565): Renable in LLVM 13
  /*
  _impl->timing_manager.setEnabled(true);
  _impl->root_timer_ = _impl->timing_manager.getRootTimer();
  _impl->tracer_timer_ = _impl->root_timer.nest("PytorchTracingTime");
  */
}

void PoptorchCompiler::endTraceTiming() {
  _impl->timing_manager.stop();

  // TODO(T49565): Renable in LLVM 13
  // We have to manually stop it anyway.
  // _impl->tracer_timer.stop();
}

void PoptorchCompiler::getTimingInfo() {
  std::cout << _impl->timing_manager.str(0) << std::endl;
  // TODO(T49565): Renable in LLVM 13
  // _impl->timing_manager.dumpAsTree();
}

void PoptorchCompiler::dump() { _impl->the_module.dump(); }

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

} // namespace poptorch_ir
