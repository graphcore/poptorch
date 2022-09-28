// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_REMOVE_UNUSED_OPERATIONS_PASS_HPP_
#define POPTORCH_REMOVE_UNUSED_OPERATIONS_PASS_HPP_

#include <memory>

namespace mlir {
template <typename T> class OperationPass;
class ModuleOp;
class PassManager;
} // namespace mlir

namespace poptorch_ir {

class CompilerContext;

/*
 * PyTorch has operations in the form: out = op(in, ..., out_storage), where
 * `out_storage` is the memory location to store the result of the op. The
 * returned `out` is a view or direct alias of that memory. In order to maintain
 * PyTorch legality we lower to this code:
 *
 *
 * %1 = EmptyTensor() # Create storage
 * %2 = perform_operation(...) # Run the op
 * %3 = copy_(%2, %1) # Copy into the correct storage location.
 *
 *
 * This is legal and correct, however it may be inefficient since the
 * EmptyTensor can just be an alias of %2.
 *
 * This pass identifies the above chain and folds it.
 */
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRemoveOverwritePass();

/*
 * This pass converts the reference semantics view operations into value
 * semantics outplace operations.
 *
 * For example, the following function
 *
 * %y = reshape(%x, ...)
 * ...[other ops]...
 * %1 = add(%y, ...)
 * overwrite replace %y with %1
 *
 * will be converted to
 *
 * ...[other ops]...
 * %1 = reshapeOutplace(%x, ...)
 * %2 = add(%1, ...)
 * %3 = reshapeInverse(%2, %x, ...)
 * overwrite replace %x with %3
 */
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOutplaceViewOpsPass();

// Add all the passes for handling overwrite ops. After these passes have been
// added, there will be no overwrite ops in the graph
void addOverwriteHandlingPasses(mlir::PassManager &manager);

} // namespace poptorch_ir

#endif // POPTORCH_REMOVE_UNUSED_OPERATIONS_PASS_HPP_
