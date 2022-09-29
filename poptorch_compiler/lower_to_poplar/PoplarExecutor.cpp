// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "lower_to_poplar/PoplarExecutor.hpp"

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/Timing.h>

#include <utility>

#include <popdist/backend.hpp>
#include <poplar/Device.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>

#include "CompilerHelpers.hpp"
#include "lower_to_poplar/IMLIRGraphConverter.hpp"
#include "lower_to_poplar/NonRestartingMLIRTimer.hpp"
#include "lower_to_poplar/PoplarDeviceAndTarget.hpp"
#include "passes/CommonPasses.hpp"
#include "passes/LowerToPoplar.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {
namespace detail {

class PoplarExecutableImpl {
public:
  explicit PoplarExecutableImpl(std::unique_ptr<poplar::Engine> engine);

  std::unique_ptr<poplar::Engine> engine;

  // Keep a reference to the buffers which are currently connected to
  // Poplar callbacks.
  std::map<std::string, Buffer> owned_buffers;
};

PoplarExecutableImpl::PoplarExecutableImpl(std::unique_ptr<poplar::Engine> e)
    : engine(std::move(e)) {}

} // namespace detail

namespace {
class MLIRToPoplarConverter final : public IMLIRGraphConverter {
public:
  explicit MLIRToPoplarConverter(CompilerContext &context)
      : _context(context) {}

protected:
  void addCustomPasses(mlir::PassManager &manager) override {
    manager.addPass(createLowerToPoplarPass(_context));
  }

private:
  CompilerContext &_context;
};

} // namespace

void PoplarExecutor::load(const PoplarDevice &device) {
  _impl->engine->load(device.device());
}

void PoplarExecutor::connectStream(const std::string &string, Buffer ptr) {
  _impl->owned_buffers.insert_or_assign(string, ptr);
  _impl->engine->connectStream(string, ptr->data());
}

void PoplarExecutor::connectStream(const std::string &string, void *ptr) {
  _impl->owned_buffers.erase(string);
  _impl->engine->connectStream(string, ptr);
}

void PoplarExecutor::execute() {
  // Use a synchronized engine run in distributed environments in order to
  // prevent host sync timeouts. It defaults to non-synchronized, regular
  // `engine.run(...)` in non-distributed environments.
  popdist::run(*_impl->engine, Programs::MainGraph);
}

void PoplarExecutor::weightsToDevice() {
  poptorch::logging::trace("Copying weights to device");
  // Use a synchronized engine run in distributed environments in order to
  // prevent host sync timeouts. It defaults to non-synchronized, regular
  // `engine.run(...)` in non-distributed environments.
  popdist::run(*_impl->engine, Programs::WeightsToDevice);
}

void PoplarExecutor::weightsToHost() {
  poptorch::logging::trace("Copying weights to host");
  // Use a synchronized engine run in distributed environments in order to
  // prevent host sync timeouts. It defaults to non-synchronized, regular
  // `engine.run(...)` in non-distributed environments.
  popdist::run(*_impl->engine, Programs::WeightsToHost);
}

PoplarExecutor::PoplarExecutor(std::unique_ptr<poplar::Engine> engine) {
  _impl = std::make_unique<detail::PoplarExecutableImpl>(std::move(engine));
}

PoplarExecutor::PoplarExecutor(PoplarExecutor &&other) {
  _impl = std::move(other._impl);
  other._impl = nullptr;
}

PoplarExecutor::~PoplarExecutor() {}

PoplarExecutor compileExecutable(mlir::ModuleOp module,
                                 const PoplarTarget &target,
                                 NonRestartingMLIRTimer &timer) {
  // The graph and sequence need to be stored outside the compiler context
  // because for PopIT we create a context inside each op handler but we
  // want them to share the graph and sequence.
  poplar::Graph graph(target.target());
  poplar::program::Sequence seq;
  CompilerContext context(graph, seq);

  MLIRToPoplarConverter converter(context);
  converter.convertGraph(module, timer);

  auto compile_poplar = timer.nestAndScope("Compiling poplar");
  auto engine =
      std::make_unique<poplar::Engine>(context.graph, context.programs);
  compile_poplar.stop();
  return PoplarExecutor(std::move(engine));
}
} // namespace poptorch_ir
