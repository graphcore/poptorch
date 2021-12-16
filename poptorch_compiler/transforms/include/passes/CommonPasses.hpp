// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_REMOVE_UNUSED_OPERATIONS_PASS_HPP_
#define POPTORCH_REMOVE_UNUSED_OPERATIONS_PASS_HPP_

#include <memory>

namespace mlir {
template <typename T> class OperationPass;
class ModuleOp;
} // namespace mlir

namespace poptorch_ir {

class CompilerContext;

/*
 * PyTorch has operations in the form of, out = op(in, ..., out_storage). Where
 * `out_storage` is the memory location to store the result of op in. The
 * returned `out` is a view or direct alias of that memory. In order to maintain
 * pytorch legality we lower to this code:
 *
 *
 * %1 = EmptyTensor() # Create storage
 * %2 = perfom_operation(...) # Run the op
 * copy_(%2, %1) # Copy into the correct storage location.
 *
 *
 * This is legal and correct, however it may be inefficient since the
 * EmptyTensor can just be an alias of %2.
 */
// This pass identifies the above chain and folds it.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRemoveRedundantCopiesPass();

// Check if an operation is unused and remove it if so.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRemoveUnusedOperationsPass();

} // namespace poptorch_ir

#endif // POPTORCH_REMOVE_UNUSED_OPERATIONS_PASS_HPP_
