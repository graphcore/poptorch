// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "lower_to_poplar/PoplarExecutor.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <utility>

#include <model_runtime/DeviceManager.hpp>
#include <passes/CommonPasses.hpp>
#include <poplar/Device.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Target.hpp>
#include <poprithms/logging/timepartitionlogger.hpp>

#include "lower_to_poplar/CompilerHelpers.hpp"
#include "passes/LowerToPoplar.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"
#include "pytorch_bridge/CompilerTypes.hpp"

namespace poptorch_ir {
namespace detail {

namespace {

// Implementation of LLVM's ostream which prints
// to poptorch::logging::trace
class LLVMStreamToTrace : public llvm::raw_ostream {
public:
  LLVMStreamToTrace() {
    // Don't buffer otherwise the calls might be printed out of order:
    // for example if the LLVM stream is used to print the graph before
    // and after passes but the pass prints messages using poptorch::logging.
    SetUnbuffered();
  }
  void write_impl(const char *ptr, size_t size) override {
    for (size_t i = 0; i < size; ++i, ++ptr) {
      // If we have an entire line: print it.
      if (*ptr == '\n') {
        poptorch::logging::trace("{}", _buf);
        _buf.clear();
      } else {
        _buf += *ptr;
      }
    }
    _pos += size;
  }

  uint64_t current_pos() const override { return _pos; }

private:
  uint64_t _pos;
  std::string _buf;
};

} // namespace

class PoplarExecutableImpl {
public:
  PoplarExecutableImpl(mlir::ModuleOp op,
                       std::shared_ptr<model_runtime::Device> device);

  void compile(poprithms::logging::ManualTimePartitionLogger &timer);

  void execute();

  std::shared_ptr<model_runtime::Device> device;

  poplar::Graph the_graph;

  mlir::ModuleOp module;

  CompilerContext context;

  std::unique_ptr<poplar::Engine> engine;

  // Keep a reference to the buffers which are currently connected to
  // Poplar callbacks.
  std::map<std::string, Buffer> owned_buffers;
};

PoplarExecutableImpl::PoplarExecutableImpl(
    mlir::ModuleOp op, std::shared_ptr<model_runtime::Device> d)
    : device(std::move(d)), the_graph(device->device().getTarget()), module(op),
      context(the_graph) {}

void PoplarExecutableImpl::execute() { engine->run(Programs::MainGraph); }

void PoplarExecutableImpl::compile(
    poprithms::logging::ManualTimePartitionLogger &timer) {
  mlir::PassManager manager{module.getContext()};

  manager.enableTiming();
  // Disable MLIR pass verification as we have our own definition of
  // valid IR state
  manager.enableVerifier(false);
  LLVMStreamToTrace output;

  // If Poptorch's TRACE logging level is enabled then print the graph
  // in between passes.
  if (poptorch::logging::shouldLog(poptorch::logging::Level::Trace)) {
    // LLVM's printing doesn't work if multi-threading is enabled.
    module.getContext()->disableMultithreading();
    mlir::OpPrintingFlags flags{};
    // enableDebugInfo = add location() at the end of each line.
    // pretty = true -> Print the actual filename:line:col rather than loc0,
    // loc1, etc which are IDs in the mlir::SourceManager.
    flags.enableDebugInfo(/* prettyForm=*/true);
    manager.enableIRPrinting([](mlir::Pass * /*unused*/,
                                mlir::Operation * /*unused*/) { return true; },
                             [](mlir::Pass * /*unused*/,
                                mlir::Operation * /*unused*/) { return true; },
                             /*printModuleScope =*/true,
                             /* printAfterOnlyOnChange =*/true, output, flags);
  }

  manager.addPass(mlir::createCanonicalizerPass());
  // TODO(T61603) Figure out why MLIR's DCE pass doesn't do the same as our
  // RemoveUnusedOperationsPass.
  // manager.addPass(mlir::createSymbolDCEPass());
  manager.addPass(poptorch_ir::createRemoveUnusedOperationsPass());
  manager.addPass(poptorch_ir::createLowerToPoplarPass(the_graph, context));

  timer.start("Poplar graph construction");
  if (mlir::succeeded(manager.run(module))) {
    timer.stop();
    timer.start("Compiling poplar");
    engine = std::make_unique<poplar::Engine>(the_graph, context.programs);
    engine->load(device->device());
    timer.stop();
  } else {
    ERROR("One or more passes failed.");
  }
}

} // namespace detail

void PoplarExecutable::connectStream(const std::string &string, Buffer ptr) {
  _impl->owned_buffers.insert_or_assign(string, ptr);
  _impl->engine->connectStream(string, ptr->data());
}

void PoplarExecutable::connectStream(const std::string &string, void *ptr) {
  _impl->owned_buffers.erase(string);
  _impl->engine->connectStream(string, ptr);
}

void PoplarExecutable::execute() { _impl->execute(); }

void PoplarExecutable::compile(
    poprithms::logging::ManualTimePartitionLogger &timer) {
  _impl->compile(timer);
}

void PoplarExecutable::weightsToDevice() {
  _impl->engine->run(poptorch_ir::Programs::WeightsToDevice);
}

void PoplarExecutable::weightsToHost() {
  _impl->engine->run(poptorch_ir::Programs::WeightsToHost);
}

void PoplarExecutable::init(mlir::ModuleOp /*unused*/) {}

PoplarExecutable::PoplarExecutable(mlir::ModuleOp module) {
  std::shared_ptr<model_runtime::Device> device;
  model_runtime::DeviceManager manager;

  bool model_enabled = false;

  // Run on model if the env var is set.
  if (const char *env_use_model = std::getenv("POPTORCH_IPU_MODEL")) {
    model_enabled = std::stoi(env_use_model) != 0;
    if (model_enabled) {
      device = manager.createIpuModelDevice(1);
    }
  }
  if (const char *env_use_model = std::getenv("POPTORCH_SMALL_IPU_MODEL")) {
    model_enabled = std::stoi(env_use_model) != 0;
    if (!device && model_enabled) {
      device = manager.createSmallIpuModelDevice(1);
    }
  }
  // Run on an actual device.
  if (!device) {
    device = manager.getDevice(1);
  }
  ERROR_ON_MSG(!device, "Failed to acquire a device");

  _impl = std::make_unique<detail::PoplarExecutableImpl>(module, device);
}

PoplarExecutable::PoplarExecutable(PoplarExecutable &&other) {
  _impl = std::move(other._impl);
  other._impl = nullptr;
}

PoplarExecutable::~PoplarExecutable() {}

} // namespace poptorch_ir
