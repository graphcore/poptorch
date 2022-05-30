// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "lower_to_poplar/PoplarExecutor.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/Timing.h>
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
  explicit PoplarExecutableImpl(std::unique_ptr<poplar::Engine> engine);

  std::unique_ptr<poplar::Engine> engine;

  // Keep a reference to the buffers which are currently connected to
  // Poplar callbacks.
  std::map<std::string, Buffer> owned_buffers;
};

PoplarExecutableImpl::PoplarExecutableImpl(std::unique_ptr<poplar::Engine> e)
    : engine(std::move(e)) {}

} // namespace detail

void PoplarExecutor::load(const poplar::Device &device) {
  _impl->engine->load(device);
}

void PoplarExecutor::connectStream(const std::string &string, Buffer ptr) {
  _impl->owned_buffers.insert_or_assign(string, ptr);
  _impl->engine->connectStream(string, ptr->data());
}

void PoplarExecutor::connectStream(const std::string &string, void *ptr) {
  _impl->owned_buffers.erase(string);
  _impl->engine->connectStream(string, ptr);
}

void PoplarExecutor::execute() { _impl->engine->run(Programs::MainGraph); }

void PoplarExecutor::weightsToDevice() {
  _impl->engine->run(Programs::WeightsToDevice);
}

void PoplarExecutor::weightsToHost() {
  _impl->engine->run(Programs::WeightsToHost);
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
                                 const poplar::Target &target,
                                 mlir::TimingScope &timer) {
  mlir::PassManager manager{module.getContext()};

  // The graph and sequence need to be stored outside the compiler context
  // because for PopIT we create a context inside each op handler but we
  // want them to share the graph and sequence.
  poplar::Graph graph(target);
  poplar::program::Sequence seq;
  CompilerContext context(graph, seq);

  auto graph_construction = timer.nest("Poplar graph construction");
  manager.enableTiming(graph_construction);
  // Disable MLIR pass verification as we have our own definition of
  // valid IR state
  manager.enableVerifier(false);
  detail::LLVMStreamToTrace output;

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
                             /* printAfterOnlyOnChange =*/true,
                             /* printAfterOnlyOnFailure =*/true, output, flags);
  }

  manager.addPass(mlir::createCanonicalizerPass());
  // TODO(T61603) Figure out why MLIR's DCE pass doesn't do the same as our
  // RemoveUnusedOperationsPass.
  // manager.addPass(mlir::createSymbolDCEPass());
  manager.addPass(createRemoveUnusedOperationsPass());
  manager.addPass(createLowerToPoplarPass(context));

  if (mlir::succeeded(manager.run(module))) {
    graph_construction.stop();
    auto compile_poplar = timer.nest("Compiling poplar");
    auto engine =
        std::make_unique<poplar::Engine>(context.graph, context.programs);
    compile_poplar.stop();
    return PoplarExecutor(std::move(engine));
  }
  ERROR("One or more passes failed.");
}
} // namespace poptorch_ir
