// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_CODEGEN_POPTORCH_IR_H_
#define POPTORCH_CODEGEN_POPTORCH_IR_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

/*
 * Custom traits.
 */

// TODO(T49565) in next LLVM version (Current 12.0) we can move this into our
// own namespace
namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class ViewOp : public TraitBase<ConcreteType, ViewOp> {
public:
  // Return the value this operation is a view of.
  mlir::Value isViewOf() const;
};

template <typename ConcreteType>
class NotImplementedOp : public TraitBase<ConcreteType, NotImplementedOp> {};

} // namespace OpTrait
} // namespace mlir

namespace poptorch_ir {

struct CompilerContext;

// Add our interfaces. These allow us to call poptorch specific functions on
// generic ops.
#include "dialect/PoptorchInterfaces.h.inc"

} // namespace poptorch_ir

// Include the operation definition.
#define GET_OP_CLASSES
#include "dialect/Poptorch.h.inc"

#endif // POPTORCH_CODEGEN_POPTORCH_IR_H_
