// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_INTERFACES_TD_
#define POPTORCH_INTERFACES_TD_

include "mlir/IR/OpBase.td"


def Poptorch_PoplarImplInterface : OpInterface<"PoplarImplInterface"> {
  let description = [{
    Any operation with this interface can be lowered to poplar. The definition must be linked
    into lower_to_poplar.
  }];

  let methods = [
    InterfaceMethod<
      [{Lowers this operation to a poplar implementation.}],
        "void",
        "lowerToPoplar",
        (ins "poptorch_ir::CompilerContext&":$context)
    >,
  ];
}

//class Poptorch_ViewInterface<string outplaceCppOpType, string inverseCppOpType>
def Poptorch_ViewInterface
     : OpInterface<"ViewInterface"> {
  let description = [{
    An op interface for view operations. Operations that implement this can be converted
    from having reference (view) semantics to having value semantics.
  }];
  // TODO: move all this to the poptorch_ir namespace. Currently we include this
  // file directly in the poptorch_ir namespace which isn't the "mlir" way of
  // doing things
  //let cppNamespace = "::poptorch_ir";
  let methods = [
    InterfaceMethod<
     /*desc=*/[{
        Create an outplace version of this operation.
     }],
     /*retType=*/"::mlir::Operation *",
     /*methodName=*/"createOutplace",
     /*args=*/(ins "::mlir::PatternRewriter&":$rewriter),
     /*methodBody=*/[{
        return rewriter.create<typename ConcreteOp::OutplaceOp>(
          $_self->getLoc(),
          $_self->getResultTypes(),
          $_self->getOperands(),
          $_self->getAttrs());
      }]>,
    InterfaceMethod<
     /*desc=*/[{
        Create an inverse of this operation.
     }],
     /*retType=*/"::mlir::Operation *",
     /*methodName=*/"createInverse",
     /*args=*/(ins "::mlir::PatternRewriter&":$rewriter, "::mlir::Value":$view),
     /*methodBody=*/[{
        llvm::SmallVector<::mlir::Type> results;
        for (const auto& operand : $_self->getOperands()) {
          const ::mlir::Type& operandType = operand.getType();
          if (operandType.isa<::mlir::RankedTensorType>()) {
            results.push_back(operandType);
          }
        }
        llvm::SmallVector<::mlir::Value> operands{view};
        llvm::copy($_self->getOperands(), std::back_inserter(operands));
        return rewriter.create<typename ConcreteOp::InverseOp>(
          $_self->getLoc(),
          results,
          operands,
          $_self->getAttrs());
      }]>
  ];
}

#endif // POPTORCH_INTERFACES_TD_
