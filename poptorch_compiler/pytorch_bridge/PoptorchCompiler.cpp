// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "pytorch_bridge/PoptorchCompiler.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/Timing.h>

#include <memory>
#include <utility>

#include "pytorch_bridge/CompilerOptions.hpp"
#include "pytorch_bridge/PytorchBridgeUtils.hpp"

#include "IMLIRCompiler.hpp"

namespace poptorch_ir {

namespace {

class ShapeInferenceCompiler : public detail::IMLIRCompiler {
public:
  using IMLIRCompiler::IMLIRCompiler;
  TensorId addInput(const mlir::RankedTensorType & /*input*/,
                    const char * /*name*/) override {
    ERROR("Not implemented");
  }

  TensorId addParameter(Buffer & /*ptr*/,
                        const mlir::RankedTensorType & /*parameter*/,
                        const char * /*name*/) override {
    ERROR("Not implemented");
  }

  void addOutput(TensorId /*id*/, const char * /*name*/) override {
    ERROR("Not implemented");
  }
};

} // namespace

PoptorchCompiler::PoptorchCompiler() {}

PoptorchCompiler::~PoptorchCompiler() {
  if (_impl) {
    // The timer crashes on destruction when it's enabled. It looks like it's
    // failing to print timing info
    _impl->timing_manager.setEnabled(false);
  }
}

PoptorchCompiler::PoptorchCompiler(PoptorchCompiler &&other) = default;
PoptorchCompiler &
PoptorchCompiler::operator=(PoptorchCompiler &&other) = default;

void PoptorchCompiler::init(ExecutionType /*execution_type*/,
                            CompilerBackend compiler_backend,
                            const poptorch::CompilerOptions &options) {
  ERROR_ON_MSG(compiler_backend != CompilerBackend::Poplar,
               "Only CompilerBackend::Poplar supported for now");
  _impl = std::make_unique<ShapeInferenceCompiler>(options);
}

TensorId PoptorchCompiler::addInput(const TensorType &type, const char *name) {
  const mlir::RankedTensorType tensor = _impl->getTensor(type);

  return _impl->addInput(tensor, name);
}

void PoptorchCompiler::setCurrentPythonCodeLocation(const char *filename,
                                                    std::uint64_t line,
                                                    std::uint64_t col) {
  // If the compiler isn't initialised yet: just ignore the call.
  if (_impl) {
    _impl->setLoc(filename, line, col);
  }
}

TensorId PoptorchCompiler::addParameter(Buffer &ptr, const TensorType &type,
                                        const char *name) {
  const mlir::RankedTensorType tensor = _impl->getTensor(type);

  return _impl->addParameter(ptr, tensor, name);
}

void PoptorchCompiler::addOutput(TensorId id, const char *name) {
  _impl->addOutput(id, name);
}

void PoptorchCompiler::startTraceTiming() {
  _impl->timing_manager.setEnabled(true);

  //_impl->root_timer.start();
  //_impl->tracer_timer = _impl->root_timer.nestAndScope("PytorchTracingTime");
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
  const mlir::Value val = _impl->findValue(id);
  const mlir::Type t1 = val.getType();
  return t1.cast<mlir::RankedTensorType>();
}

bool PoptorchCompiler::allOpsCanBeLoweredToPoplar() const {
  return _impl->allOpsCanBeLoweredToPoplar();
}

} // namespace poptorch_ir
