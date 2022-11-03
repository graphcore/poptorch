// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "lower_to_poplar/IMLIRGraphConverter.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/Timing.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Transforms/ViewOpGraph.h>

#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>
#include <poptorch_logging/Error.hpp>

#include "CompilerHelpers.hpp"
#include "lower_to_poplar/NonRestartingMLIRTimer.hpp"
#include "passes/CommonPasses.hpp"
#include "poptorch_logging/Logging.hpp"
#include "pytorch_bridge/CompilerTypes.hpp"

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

template <typename Range>
void filter(const mlir::BitVector &bit_vector, Range &rng) {
  auto to_remove_from = rng.end();

  // Loop backwards through the range moving removed elements to the end so
  // removals don't mess up the indexing
  for (auto itr = rng.rbegin(); itr != rng.rend(); ++itr) {
    auto index = std::distance(itr, rng.rend()) - 1;
    if (bit_vector[index]) {
      to_remove_from =
          std::rotate(std::prev(itr.base()), itr.base(), to_remove_from);
    }
  }

  rng.erase(to_remove_from, rng.end());
}

void optimizeOutputs(mlir::func::FuncOp &func,
                     const std::vector<TensorId> &inputs,
                     std::vector<TensorId> &outputs) {

  ERROR_ON_MSG(!func.getBody().hasOneBlock(),
               "We currently don't handle functions with multiple blocks");

  auto *terminator = func.getBody().getBlocks().front().getTerminator();
  auto return_op = mlir::dyn_cast_or_null<mlir::func::ReturnOp>(terminator);
  ERROR_ON(!return_op);

  mlir::BitVector unchanged_outputs(outputs.size());
  auto func_arguments = func.getArguments();
  for (const auto &result : llvm::enumerate(return_op.getOperands())) {
    if (const auto *arg = llvm::find(func_arguments, result.value());
        arg != func_arguments.end()) {
      // Only remove the output if it is being written back to the same place it
      // was read from (otherwise it's a copy which we want to preserve)
      // TODO(T71081): convert these to on device copies and remove them from
      // the outputs
      if (inputs[arg->getArgNumber()] == outputs[result.index()]) {
        unchanged_outputs.set(result.index());
      }
    }
  }

  filter(unchanged_outputs, outputs);
  return_op->eraseOperands(unchanged_outputs);
  func.eraseResults(unchanged_outputs);
}

void optimizeInputs(mlir::func::FuncOp &func, std::vector<TensorId> &inputs) {
  mlir::BitVector unused_inputs(inputs.size());
  for (const auto &argument : func.getArguments()) {
    if (argument.getUses().empty()) {
      unused_inputs.set(argument.getArgNumber());
    }
  }

  filter(unused_inputs, inputs);
  func.eraseArguments(unused_inputs);
}

void optimizeInputAndOutputs(mlir::func::FuncOp &func,
                             std::vector<TensorId> &inputs,
                             std::vector<TensorId> &outputs) {
  ERROR_ON(func.getNumArguments() != inputs.size());
  ERROR_ON(func.getNumResults() != outputs.size());

  optimizeOutputs(func, inputs, outputs);
  optimizeInputs(func, inputs);
}

} // namespace

void runGraphPasses(mlir::ModuleOp &module, ExternalFunctionIO &io,
                    NonRestartingMLIRTimer &timer) {
  ERROR_ON(mlir::verify(module).failed());

  mlir::PassManager manager{module.getContext()};

  auto graph_passes = timer.nestAndScope("MLIR graph passes");
  manager.enableTiming(graph_passes);
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
  // Note: we canonicalize before the outplace view ops pass since some views
  // are implemented in terms of other views
  manager.addPass(mlir::createCanonicalizerPass());
  addOverwriteHandlingPasses(manager);
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createCSEPass());

  ERROR_ON_MSG(!mlir::succeeded(manager.run(module)),
               "One or more passes failed.");

  for (auto &[symbol_name, function_io] : io) {
    auto func = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
        module.lookupSymbol(symbol_name));
    ERROR_ON(!func);
    optimizeInputAndOutputs(func, function_io.inputs, function_io.outputs);
  }

  ERROR_ON(mlir::verify(module).failed());
}

void IMLIRGraphConverter::convertGraph(mlir::ModuleOp &module,
                                       NonRestartingMLIRTimer &timer) {
  mlir::PassManager manager{module.getContext()};
  auto graph_construction = timer.nestAndScope("Poplar graph construction");

  manager.enableTiming(graph_construction);
  addCustomPasses(manager);

  ERROR_ON_MSG(!mlir::succeeded(manager.run(module)),
               "Converting the graph failed.");

  graph_construction.stop();
}

} // namespace poptorch_ir
