// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "pytorch_bridge/PoptorchCompiler.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/Timing.h>

#include <utility>

#include "lower_to_poplar/PoplarDeviceAndTarget.hpp"
#include "pytorch_bridge/CompilerOptions.hpp"
#include "pytorch_bridge/PytorchBridgeUtils.hpp"

#include "IMLIRCompiler.hpp"
#include "MLIREagerCompiler.hpp"
#include "MLIRStaticGraphCompiler.hpp"

namespace poptorch_ir {

PoptorchCompiler::PoptorchCompiler() {}

PoptorchCompiler::~PoptorchCompiler() {
  if (_impl) {
    // The timer crashes on destruction when it's enabled. It looks like it's
    // failing to print timing info
    _impl->timing_manager.setEnabled(false);
  }
}

void PoptorchCompiler::init(ExecutionType execution_type,
                            CompilerBackend compiler_backend,
                            const poptorch::CompilerOptions &options) {
  ERROR_ON_MSG(compiler_backend != CompilerBackend::Poplar,
               "Only CompilerBackend::Poplar supported for now");

  if (compiler_backend == CompilerBackend::Poplar) {
    if (execution_type == ExecutionType::StaticGraph) {
      _impl = std::make_unique<detail::MLIRStaticGraphCompiler>(options);
    } else if (execution_type == ExecutionType::EagerMode) {
      PoplarDevice device = PoplarDevice::defaultDevice();
      _impl = std::make_unique<detail::MLIREagerCompiler>(device, options);
    }
  }
  ERROR_ON(_impl == nullptr);
}

const poptorch::CompilerOptions &PoptorchCompiler::getOptions() const {
  return const_cast<PoptorchCompiler *>(this)->getMutableOptions();
}

poptorch::CompilerOptions &PoptorchCompiler::getMutableOptions() {
  return _impl->getMutableOptions();
}

void PoptorchCompiler::onOpAdded() { _impl->onOpAdded(); }

TensorId PoptorchCompiler::addInput(const Buffer &ptr,
                                    const std::vector<std::int64_t> &shape,
                                    Type type, const char *name) {
  mlir::RankedTensorType tensor = _impl->getTensor(type, shape);

  return _impl->addInput(ptr, tensor, name);
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

  return _impl->addParameter(ptr, tensor, name);
}

void PoptorchCompiler::addOutput(TensorId id, void *ptr, const char *name) {
  _impl->addOutput(ptr, id, name);
}

void PoptorchCompiler::startTraceTiming() {
  _impl->timing_manager.setEnabled(true);

  _impl->root_timer.start();
  _impl->tracer_timer = _impl->root_timer.nestAndScope("PytorchTracingTime");
}

void PoptorchCompiler::endTraceTiming() {
  // Stop the timer
  _impl->tracer_timer = mlir::TimingScope();
}

void PoptorchCompiler::getTimingInfo() { _impl->timing_manager.dumpAsTree(); }

void PoptorchCompiler::dump() { _impl->dump(); }

bool PoptorchCompiler::isView(poptorch_ir::TensorId id) const {
  mlir::Operation *op = _impl->findValue(id).getDefiningOp();

  return op != nullptr && op->hasTrait<mlir::OpTrait::ViewOp>();
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
  _impl->addReturn();
}

bool PoptorchCompiler::allOpsCanBeLoweredToPoplar() const {
  return _impl->allOpsCanBeLoweredToPoplar();
}

void PoptorchCompiler::compileRunAndReset() {
  auto *compiler = dynamic_cast<detail::MLIREagerCompiler *>(_impl.get());
  ERROR_ON_MSG(compiler == nullptr,
               "[Internal] Only eager builders can compileRunAndReset()");
  compiler->compileRunAndReset();
}

PoplarExecutorWrapper PoptorchCompiler::compileAndLoad() {
  auto *compiler = dynamic_cast<detail::MLIRStaticGraphCompiler *>(_impl.get());
  ERROR_ON_MSG(compiler == nullptr,
               "[Internal] Only static graph builders can compileAndLoad()");
  ERROR_ON_MSG(
      compiler->input_callbacks.empty() && compiler->output_callbacks.empty(),
      "Either no inputs or outputs were added or compiling a second time.");

  // Obtain the device
  PoplarDevice device = PoplarDevice::defaultDevice();
  auto exe = compiler->compile(device.getTarget());

  exe.load(device);

  // Move across and connect callbacks, clearing them in the process
  for (auto &pair : compiler->output_callbacks) {
    exe.connectStream(pair.first, pair.second);
  }
  compiler->output_callbacks.clear();

  for (auto &pair : compiler->weight_callbacks) {
    exe.connectStream("Write-" + pair.first, pair.second);
    exe.connectStream("Read-" + pair.first, std::move(pair.second));
  }
  compiler->weight_callbacks.clear();

  PoplarExecutorWrapper executor(std::move(exe),
                                 std::move(compiler->input_callbacks));
  compiler->input_callbacks.clear();

  return executor;
}

} // namespace poptorch_ir
