// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/PopitExecutor.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/Timing.h>
#include <mlir/Transforms/Passes.h>

#include <memory>
#include <thread>
#include <utility>

#include <passes/CommonPasses.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>

#include <popit/Device.hpp>

#include "CompilerHelpers.hpp"
#include "PopitContext.hpp"
#include "lower_to_poplar/IMLIRGraphConverter.hpp"
#include "lower_to_poplar/NonRestartingMLIRTimer.hpp"
#include "passes/LowerToPopit.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {

namespace {
class MLIRToPopitConverter final : public IMLIRGraphConverter {
public:
  explicit MLIRToPopitConverter(PopitContext &popit) : _context(popit) {}

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
      popit::malloc(context.session.get(), tensor_spec.type, numel));
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

bool shouldWaitIfIpuIsUnavailable() {
  bool wait = false;
  if (const char *env_wait_for_ipu = std::getenv("POPTORCH_WAIT_FOR_IPU")) {
    wait = std::stoi(env_wait_for_ipu) != 0;
    poptorch::logging::info(
        "From POPTORCH_WAIT_FOR_IPU environment variable: If no IPU "
        "is available: {}",
        wait ? "Wait" : "Fail & exit");
  }
  return wait;
}

popit::Device getPopitDevice() {
  bool model_enabled = false;
  if (const char *env_use_model = std::getenv("POPTORCH_IPU_MODEL")) {
    model_enabled = std::stoi(env_use_model) != 0;
    ERROR_ON_MSG(model_enabled, "IPU model is unsupported in eager mode");
  }
  if (const char *env_use_model = std::getenv("POPTORCH_SMALL_IPU_MODEL")) {
    model_enabled = std::stoi(env_use_model) != 0;
    ERROR_ON_MSG(model_enabled, "IPU model is unsupported in eager mode");
  }

  // Otherwise attempt to acquire hardware
  popit::DeviceManager device_manager;
  auto devices = device_manager.getDevices(/*requiredNumIpus=*/1);
  if (devices.empty()) {
    ERROR("No devices found");
  }

  bool wait_for_ipu = shouldWaitIfIpuIsUnavailable();
  do {
    for (auto &device : devices) {
      if (device.attach()) {
        return std::move(device);
      }
    }
    if (wait_for_ipu) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  } while (wait_for_ipu);
  ERROR("Failed to attach to any of the IPU devices.");
  return popit::Device();
}

PopitContext::PopitContext()
    : session(nullptr,
              [](popit::Session *) { /* nothing to do for nullptr */ }),
      device(getPopitDevice()) {}

PopitContext::~PopitContext() {
  tensors.clear();
  session.reset();
}

PopitExecutor::PopitExecutor() : _context(std::make_unique<PopitContext>()) {
  poplar::Target target = _context->device.getTarget();

  _context->session =
      std::unique_ptr<popit::Session_t, decltype(&popit::destroySession)>(
          popit::createSession(&target),
          [](popit::Session *s) { popit::destroySession(s); });
  popit::connect(_context->session.get(), &_context->device);
}

PopitExecutor::~PopitExecutor() {}

void PopitExecutor::addInput(const Buffer &ptr,
                             const mlir::RankedTensorType &input, TensorId id) {
  PopitMemPtr tensor = allocatePopitTensor(*_context, input);
  popit::copyFromHost(ptr->data(), tensor.get());
  ERROR_ON(!_context->tensors.try_emplace(id, tensor).second);
}

void PopitExecutor::readOutput(TensorId id, void *ptr) {
  auto it = _context->tensors.find(id);
  ERROR_ON_MSG(it == _context->tensors.end(), "Unknown tensor ID " << id);
  poptorch::logging::trace("Copying tensor {} to host pointer {}", id, ptr);
  popit::copyToHost(it->second.get(), reinterpret_cast<char *>(ptr));
}

void PopitExecutor::freeTensor(TensorId id) { _context->tensors.erase(id); }

void PopitExecutor::compileAndRun(
    mlir::ModuleOp module, NonRestartingMLIRTimer &timer,
    const llvm::DenseMap<mlir::Value, TensorId> &mappings) {

  if (!containsPoplarOps(module)) {
    poptorch::logging::trace("MLIR graph empty: skipping compileAndRun()");
    return;
  }

  auto compile_popit = timer.nestAndScope("Compiling popit");

  MLIRToPopitConverter converter(*_context);
  converter.convertGraph(module, timer);
  compile_popit.stop();

  auto run_popit = timer.nestAndScope("Executing popit");

  // Map graph input values to PopIT memory pointers.
  std::vector<popit::Mem_t *> inputs;
  inputs.reserve(_context->inputs.size());
  for (auto &input : _context->inputs) {
    auto it = mappings.find(input);
    // This can only happen if a pass in MLIRToPopitConverter replaces one of
    // the graph inputs. We can't support this because we've got no way to map
    // the new graph input to a Torch tensor.
    ERROR_ON_MSG(it == mappings.end(),
                 "[Internal] Input Value not found in tensor map");
    inputs.push_back(_context->tensors.at(it->second).get());
  }

  // Execute the function
  std::vector<popit::Mem_t *> outputs =
      popit::call(_context->session.get(), _context->popit_fn,
                  /* ipuIndex=*/0, inputs);

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
