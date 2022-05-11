// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "passes/CommonPasses.hpp"
#include "passes/PassUtils.hpp"

#include "dialect/PoptorchDialect.hpp"
#include "poptorch_logging/Error.hpp"
#include "poptorch_logging/Logging.hpp"

namespace poptorch_ir {

namespace {
/*
  Converts the MLIR graph into a poplar graph which can then be compiled.
 */
class RemoveUnusedOperations
    : public mlir::PassWrapper<RemoveUnusedOperations,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  RemoveUnusedOperations() = default;

  void runOnOperation() override;
};

void RemoveUnusedOperations::runOnOperation() {
  mlir::ModuleOp module = this->getOperation();

  std::vector<mlir::Operation *> to_remove;

  mlir::FuncOp main_graph = *module.getOps<mlir::FuncOp>().begin();
  ERROR_ON(main_graph.getName().str() != "MainGraph");
  for (mlir::Operation &op : main_graph.getOps()) {
    // Check that number of results > 0 so we don't delete in-place
    // operations
    if (op.getNumResults() > 0 && op.use_empty()) {
      to_remove.push_back(&op);
    }
  }
  for (auto *op : to_remove) {
    op->erase();
  }
}

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRemoveUnusedOperationsPass() {
  return std::make_unique<RemoveUnusedOperations>();
}

} // namespace poptorch_ir

static mlir::PassRegistration<poptorch_ir::RemoveUnusedOperations>
    remove_unused_operations("remove-unused-operations", "");
