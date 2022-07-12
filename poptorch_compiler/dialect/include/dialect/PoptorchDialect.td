// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPTORCH_IR_DIALECT
#define POPTORCH_IR_DIALECT

include "mlir/IR/OpBase.td"
include "PoptorchInterfaces.td"

def Poptorch_Dialect : Dialect {
    let name = "poptorch";

    let summary = "PopTorch MLIR target";

    let description = [{
        This dialect sits inbetween canonicalised torch::jit IR and Poplar/poplibs.
    }];


    let cppNamespace = "::poptorch_ir";
}

def ViewOp : NativeOpTrait<"ViewOp">;
// Ops with this trait implement shape inference but can't be lowered to
// Poplar (yet).
def NotImplementedOp : NativeOpTrait<"NotImplementedOp">;

// Mark an op as implicit casting on the specified index
// All indices marked in this way will have their operands cast to a
// common type just before the op is called.
class ImplicitCastOperand<int idx>
  : ParamNativeOpTrait<"ImplicitCastOperand", !cast<string>(idx)>;

def ImplicitCastToBool : NativeOpTrait<"ImplicitCastToBool">;
def ImplicitCastToFloat : NativeOpTrait<"ImplicitCastToFloat">;

// All operations coming through the dispatcher's fallback should inherit
// from either Poptorch_Op if they can be lowered to Poplar or
// Poptorch_NotImplementedOp if they cannot.
class Poptorch_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Poptorch_Dialect, mnemonic, traits # [DeclareOpInterfaceMethods<Poptorch_PoplarImplInterface>]>;

class Poptorch_NotImplementedOp<string mnemonic, list<OpTrait> traits = []> :
        Op<Poptorch_Dialect, mnemonic, traits # [NotImplementedOp]>;

class Poptorch_AbstractOp<string mnemonic, list<OpTrait> traits = []> :
        Op<Poptorch_Dialect, mnemonic, traits>;


def valid_ints : SignedIntOfWidths<[8, 16, 32]>;
def valid_uints : UnsignedIntOfWidths<[8]>;
def valid_float : FloatOfWidths<[16, 32]>;

def valid_tensor : StaticShapeTensorOf<[UI1, valid_ints, valid_uints, valid_float]>;
def valid_float_tensor : StaticShapeTensorOf<[valid_float]>;
def valid_integral_tensor : StaticShapeTensorOf<[UI1, valid_ints, valid_uints]>;
def valid_non_boolean_tensor : StaticShapeTensorOf<[valid_ints, valid_uints, valid_float]>; 

/*
    We will add in specific types for parameters and buffers later.
*/
// Any valid tensor type.
def Poptorch_tensor : AnyTypeOf<[valid_tensor]>;
def Poptorch_float_tensor : AnyTypeOf<[valid_float_tensor]>;
def Poptorch_integral_tensor : AnyTypeOf<[valid_integral_tensor]>;
def Poptorch_non_boolean_tensor : AnyTypeOf<[valid_non_boolean_tensor]>;

def Poptorch_tensor_no_grad : AnyTypeOf<[valid_tensor]>;

def Poptorch_tensorlist : Variadic<Poptorch_tensor>;

#endif // POPTORCH_IR_DIALECT
