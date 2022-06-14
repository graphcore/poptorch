// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_CODEGEN_POPTORCH_IR_H_
#define POPTORCH_CODEGEN_POPTORCH_IR_H_

#include <algorithm>
#include <numeric>
#include <vector>

#include "dialect/Helpers.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

/*
 * Custom traits.
 */

// TODO(T62821) in next LLVM version (Current 13.0) we can move this into our
// own namespace. Note: this comment was also present for LLVM 13.0 so remove it
// if we can't do this in LLVM 14
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

// Used for logical and/or/xor to force casting to a bool before running the
// operation
template <typename ConcreteType>
class ImplicitCastToBool : public TraitBase<ConcreteType, ImplicitCastToBool> {
};

// Used for division to casting to a float before running the operation
template <typename ConcreteType>
class ImplicitCastToFloat
    : public TraitBase<ConcreteType, ImplicitCastToFloat> {};

const size_t max_implicit_casting_operands = 3;
template <unsigned idx> class ImplicitCastOperand {
public:
  static_assert(idx < max_implicit_casting_operands);

  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, ImplicitCastOperand<idx>::Impl> {
  };
};

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
