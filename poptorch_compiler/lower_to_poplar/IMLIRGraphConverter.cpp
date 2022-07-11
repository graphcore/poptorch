// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/IMLIRGraphConverter.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/Timing.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Transforms/ViewOpGraph.h>

#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>

#include "CompilerHelpers.hpp"
#include "lower_to_poplar/NonRestartingMLIRTimer.hpp"
#include "passes/CommonPasses.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {

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

void IMLIRGraphConverter::convertGraph(mlir::ModuleOp &module,
                                       NonRestartingMLIRTimer &timer) {
  mlir::PassManager manager{module.getContext()};

  auto graph_construction = timer.nestAndScope("Poplar graph construction");
  manager.enableTiming(graph_construction);
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
                             /* printAfterOnlyOnChange =*/true,
                             /* printAfterOnlyOnFailure =*/true, output, flags);
  }

  // fd needs to remain open until manager.run() has been called.
  std::unique_ptr<llvm::raw_fd_ostream> fd;
  if (const char *dot_filename = std::getenv("POPTORCH_MLIR_DOT_FILE")) {
    std::error_code err;
    fd = std::make_unique<llvm::raw_fd_ostream>(dot_filename, err);
    ERROR_ON_MSG(err,
                 "Failed to open " << dot_filename << ": " << err.message());
    manager.addPass(mlir::createPrintOpGraphPass(*fd));
  }
  manager.addPass(mlir::createCanonicalizerPass());
  // TODO(T61603) Figure out why MLIR's DCE pass doesn't do the same as our
  // RemoveUnusedOperationsPass.
  // manager.addPass(mlir::createSymbolDCEPass());
  manager.addPass(createRemoveUnusedOperationsPass());
  addCustomPasses(manager);

  ERROR_ON_MSG(!mlir::succeeded(manager.run(module)),
               "One or more passes failed.");
  graph_construction.stop();
}

} // namespace poptorch_ir
