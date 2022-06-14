// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/PopitExecutor.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/Timing.h>
#include <mlir/Transforms/Passes.h>

#include <memory>
#include <utility>

#include <model_runtime/DeviceManager.hpp>
#include <passes/CommonPasses.hpp>
#include <poplar/Device.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>

#include "CompilerHelpers.hpp"
#include "PopitContext.hpp"
#include "lower_to_poplar/IMlirGraphConverter.hpp"
#include "lower_to_poplar/NonRestartingMlirTimer.hpp"
#include "lower_to_poplar/PoplarDeviceAndTarget.hpp"
#include "passes/LowerToPopit.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {

namespace {
class MlirToPopitConverter final : public IMlirGraphConverter {
public:
  explicit MlirToPopitConverter(PopitContext &popit) : _context(popit) {}

protected:
  void addCustomPasses(mlir::PassManager &manager) override {
    manager.addPass(createLowerToPopitPass(_context));
  }

private:
  PopitContext &_context;
};

PopitMemPtr allocatePopitTensor(PopitContext &context,
                                const mlir::RankedTensorType &type) {
  const popit::TensorSpec tensor_spec = getTensorSpec(type);
  const popit::TensorShape tensor_shape = tensor_spec.shape;
  std::uint64_t numel =
      std::accumulate(tensor_shape.begin(), tensor_shape.end(), 1,
                      std::multiplies<std::uint64_t>{});

  return PopitMemPtr(
      popitMalloc(context.session.get(), tensor_spec.type, numel));
}

bool containsPoplarOps(mlir::ModuleOp &module) {
  bool found_op = false;
  module.walk([&](PoplarImplInterface) {
    found_op = true;
    return mlir::WalkResult::interrupt();
  });
  return found_op;
}
} // namespace

PopitContext::PopitContext()
    : session(nullptr, [](popitSession *) { /* nothing to do for nullptr */ }) {
}

PopitContext::~PopitContext() {
  tensors.clear();
  session.reset();
}

PopitExecutor::PopitExecutor(PoplarDevice &device)
    : _context(std::make_unique<PopitContext>()) {
  const poplar::Target &target = device.device().getTarget();
  _context->session =
      std::unique_ptr<popitSession_t, decltype(&popitDestroySession)>(
          popitCreateSession(&target), &popitDestroySession);
  popitConnect(_context->session.get(), &device.device());
}

PopitExecutor::~PopitExecutor() {}

void PopitExecutor::addInput(const Buffer &ptr,
                             const mlir::RankedTensorType &input, TensorId id) {
  PopitMemPtr tensor = allocatePopitTensor(*_context, input);
  popitCopyFromHost(ptr->data(), tensor.get());
  ERROR_ON(!_context->tensors.try_emplace(id, tensor).second);
}

void PopitExecutor::readOutput(TensorId id, void *ptr) {
  auto it = _context->tensors.find(id);
  ERROR_ON_MSG(it == _context->tensors.end(), "Unknown tensor ID " << id);
  poptorch::logging::trace("Copying tensor {} to host pointer {}", id, ptr);
  popitCopyToHost(it->second.get(), reinterpret_cast<char *>(ptr));
}

void PopitExecutor::freeTensor(TensorId id) { _context->tensors.erase(id); }

void PopitExecutor::compileAndRun(
    mlir::ModuleOp module, NonRestartingMlirTimer &timer,
    const llvm::DenseMap<mlir::Value, TensorId> &mappings) {

  if (!containsPoplarOps(module)) {
    poptorch::logging::trace("MLIR graph empty: skipping compileAndRun()");
    return;
  }

  auto compile_popit = timer.nestAndScope("Compiling popit");

  MlirToPopitConverter converter(*_context);
  converter.convertGraph(module, timer);
  compile_popit.stop();

  auto run_popit = timer.nestAndScope("Executing popit");

  // Map graph input values to PopIT memory pointers.
  std::vector<popitMem_t *> inputs;
  inputs.reserve(_context->inputs.size());
  for (auto &input : _context->inputs) {
    auto it = mappings.find(input);
    // This can only happen if a pass in MlirToPopitConverter replaces one of
    // the graph inputs. We can't support this because we've got no way to map
    // the new graph input to a Torch tensor.
    ERROR_ON_MSG(it == mappings.end(),
                 "[Internal] Input Value not found in tensor map");
    inputs.push_back(_context->tensors.at(it->second).get());
  }

  // Execute the function
  std::vector<popitMem_t *> outputs =
      popitCall(_context->session.get(), _context->popit_fn, /* ipuIndex=*/0,
                /* ipuMemIndex=*/0, inputs);

  // Map PopIT memory pointers to graph outputs.
  ERROR_ON(outputs.size() != _context->output_ids.size());
  for (std::uint64_t i = 0; i < outputs.size(); i++) {
    auto id = _context->output_ids.at(i);
    auto out = _context->tensors.find(id);
    if (out == _context->tensors.end()) {
      _context->tensors.try_emplace(id, PopitMemPtr(outputs.at(i)));
    } else {
      out->second = PopitMemPtr(outputs.at(i));
    }
  }
  run_popit.stop();
}
} // namespace poptorch_ir
