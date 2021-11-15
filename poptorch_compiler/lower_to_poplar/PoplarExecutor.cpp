// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <model_runtime/DeviceManager.hpp>
#include <poplar/Device.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Target.hpp>
#include <poprithms/logging/timepartitionlogger.hpp>

#include "poptorch_logging/Logging.hpp"
#include "poptorch_logging/Error.hpp"
#include "lower_to_poplar/CompilerHelpers.hpp"
#include "lower_to_poplar/LowerToPoplar.hpp"
#include "lower_to_poplar/PoplarExecutor.hpp"


namespace poptorch_ir {
namespace detail {

class PoplarExecutableImpl {
public:

  PoplarExecutableImpl(mlir::ModuleOp op, std::shared_ptr<model_runtime::Device>device);

  void compile(poprithms::logging::ManualTimePartitionLogger &timer);

  void execute();

  std::shared_ptr<model_runtime::Device> device;

  poplar::Graph the_graph;

  mlir::ModuleOp module;

  CompilerContext context;

  std::unique_ptr<poplar::Engine> engine;
};

PoplarExecutableImpl::PoplarExecutableImpl(mlir::ModuleOp op,
                                           std::shared_ptr<model_runtime::Device> d)
    : device(d), the_graph(device->device().getTarget()), module(op),
      context(the_graph) {}

void PoplarExecutableImpl::execute() { engine->run(Programs::MainGraph); }

void PoplarExecutableImpl::compile(
    poprithms::logging::ManualTimePartitionLogger &timer) {
  mlir::PassManager manager{module.getContext()};

  // manager.enableStatistics();

  timer.start("Poplar graph construction");
  manager.enableTiming();
  manager.addPass(poptorch_ir::createLowerToPoplarPass(the_graph, context));

  mlir::LogicalResult result = manager.run(module);
  (void)result;
  timer.stop();

  timer.start("Compiling poplar");
  engine = std::make_unique<poplar::Engine>(the_graph, context.programs);
  engine->load(device->device());
  timer.stop();
}

} // namespace detail

void PoplarExecutable::connectStream(const std::string &string, void *ptr) {
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

  _impl =
      std::make_unique<detail::PoplarExecutableImpl>(module, device);
}

PoplarExecutable::PoplarExecutable(PoplarExecutable &&other) {
  _impl = std::move(other._impl);
  other._impl = nullptr;
}

PoplarExecutable::~PoplarExecutable() {}

} // namespace poptorch_ir
