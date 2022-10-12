// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "LowerToPoplar.hpp"

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "../CompilerHelpers.hpp"
#include "passes/PassUtils.hpp"

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/Fill.hpp>
#include <poprand/RandomGen.hpp>

#include "dialect/PoptorchDialect.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {

namespace {
Programs getProgramType(std::string_view name) {
  if (name == "MainGraph") {
    return Programs::MainGraph;
  }
  if (name == "WeightsToDevice") {
    return Programs::WeightsToDevice;
  }
  if (name == "WeightsToHost") {
    return Programs::WeightsToHost;
  }
  ERROR("Unexpected program name " + std::string(name));
}

/*
  Converts the MLIR graph into a poplar graph which can then be compiled.
 */
class LowerToPoplar final
    : public mlir::PassWrapper<LowerToPoplar,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  LowerToPoplar() {}

  explicit LowerToPoplar(CompilerContext &c) : _context(&c) {}

  void runOnOperation() override;

  mlir::StringRef getArgument() const final { return "lower-to-poplar"; }

private:
  // Verify operations using MLIR's verifier.
  static void verifyOperations(const mlir::func::FuncOp &function);

  CompilerContext *_context;
};

void LowerToPoplar::runOnOperation() {
  mlir::ModuleOp module = this->getOperation();

  poptorch::logging::info("Graph lowered to poplar:\n{}", mlirOpToStr(module));

  // Compile all the programs in the module
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    auto program = getProgramType(function.getName());
    _context->clearLocalData();

    verifyOperations(function);
    _context->seq = poplar::program::Sequence();

    // Walk over all functions with a poplar impl.
    function.walk(
        [&](PoplarImplInterface impl) { impl.lowerToPoplar(*_context); });

    _context->programs[program] = _context->seq;
  }
}

void LowerToPoplar::verifyOperations(const mlir::func::FuncOp &function) {
  // It would be possible to call mlir::verify on the whole graph, however
  // this would not pinpoint the failing operation. Therefore, we verify each
  // op at a time, by recursing into it. Note that calling mlir::verify has some
  // extra checks ommited here.
  auto num_regions = function->getNumRegions();
  for (unsigned i = 0; i < num_regions; i++) {
    auto &region = function->getRegion(i);

    for (auto &block : region) {
      for (auto &op : block) {
        std::string const op_name = op.getName().getStringRef().str();

        // WeightsToDevice/Host are FuncOps, which means that they should be
        // isolated from external variables. However, this is not the case so
        // verification would fail. There is no interface to remove a trait
        // so they would have to be defined as some other type such as a
        // custom defined non-isolated function op in the tablegen. Instead,
        // we simply skip verification. The name will be "func" when calling
        // getName on the mlir::Operation (here) rather than the
        // mlir::func::FuncOp.
        if (op_name == "func") {
          continue;
        }

        if (mlir::failed(mlir::verify(&op))) {
          ERROR("Verification failed for " << op_name << ":\n "
                                           << mlirOpToStr(op));
        }
      }
    }
  }
}
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLowerToPoplarPass(CompilerContext &context) {
  return std::make_unique<LowerToPoplar>(context);
}

} // namespace poptorch_ir

static mlir::PassRegistration<poptorch_ir::LowerToPoplar> lower;
