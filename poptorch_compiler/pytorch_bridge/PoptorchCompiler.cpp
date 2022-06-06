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

void PoptorchCompiler::init(ExecutionType execution_type,
                            CompilerBackend compiler_backend) {
  ERROR_ON_MSG(execution_type != ExecutionType::StaticGraph,
               "Only ExecutionMode::StaticGraph supported for now");
  ERROR_ON_MSG(compiler_backend != CompilerBackend::Poplar,
               "Only CompilerBackend::Poplar supported for now");
  if (execution_type == ExecutionType::StaticGraph &&
      compiler_backend == CompilerBackend::Poplar) {
    _impl = std::make_unique<detail::MLIRStaticGraphBuilder>();
  }
  ERROR_ON(_impl == nullptr);
}

TensorId PoptorchCompiler::addInput(const Buffer &ptr,
                                    const std::vector<std::int64_t> &shape,
                                    Type type, const char *name) {
  mlir::RankedTensorType tensor = _impl->getTensor(type, shape);
  // Add the argument to the graph args.
  mlir::Value val = _impl->addArgumentToMainGraph(tensor);

  _impl->addInput(ptr, val, name);

  return _impl->addValue(val);
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

  // Add the argument to the graph args.
  mlir::Value val = _impl->addArgumentToMainGraph(tensor);

  _impl->addParameter(ptr, val, name);

  return _impl->addValue(val);
}

void PoptorchCompiler::addOutput(TensorId id, void *ptr, const char *name) {
  mlir::Value val = _impl->findValue(id);
  _impl->addOutput(ptr, val, name);
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
  mlir::Operation *op = _impl->findValue(id).getDefiningOp();

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
  mlir::Value val = _impl->findValue(id);
  mlir::Type t1 = val.getType();
  return t1.cast<mlir::RankedTensorType>();
}

void PoptorchCompiler::addReturn() {
  // Add returns to each of the graphs.
  _impl->createOp<poptorch_ir::end_graph>();
  _impl->addReturn();
}

bool PoptorchCompiler::allOpsCanBeLoweredToPoplar() const {
  return _impl->allOpsCanBeLoweredToPoplar();
}

void PoptorchCompiler::compileRunAndReset() {
  auto *compiler = dynamic_cast<detail::MLIREagerBuilder *>(_impl.get());
  ERROR_ON_MSG(compiler == nullptr,
               "[Internal] Only eager builders can compileRunAndReset()");
  compiler->compileRunAndReset();
}

PoplarExecutorWrapper PoptorchCompiler::compileAndLoad() {
  auto *compiler = dynamic_cast<detail::MLIRStaticGraphBuilder *>(_impl.get());
  ERROR_ON_MSG(compiler == nullptr,
               "[Internal] Only static graph builders can compileAndLoad()");

  auto device = getDevice();
  auto exe = compiler->compile(device->device().getTarget());
  exe.load(device->device());

  // Connect up the outputs.
  for (auto &pair : compiler->output_callbacks) {
    exe.connectStream(pair.first, pair.second);
  }

  for (auto &pair : compiler->weight_callbacks) {
    exe.connectStream("Write-" + pair.first, pair.second);
    exe.connectStream("Read-" + pair.first, pair.second);
  }

  PoplarExecutorWrapper executor(std::move(exe),
                                 std::move(compiler->input_callbacks));
  compiler->weight_callbacks.clear();
  // input_callbacks was moved to PoplarExecutor in compile()
  return executor;
}

} // namespace poptorch_ir
