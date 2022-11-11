// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPTORCH_IR_DIALECT
#define POPTORCH_IR_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
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
// from Poptorch_BasicOp. The following classes are provided for convenience:
// * Poptorch_Op for ops that can be lowered to Poplar directly,
// * Poptorch_ViewOp for view operations which may be lowered (alternatively
//   you may use a pass to convert a Poptorch_AbstractOp with the ViewOp trait
//   to a Poptorch_ViewOp)
// * Poptorch_NotImplementedOp for ops that cannot currently be lowered to
//   poplar, or
// * Potorch_AbstractOp for ops that will be lower using other passes
class Poptorch_BasicOp<string mnemonic, list<Trait> traits = []> :
        Op<Poptorch_Dialect, mnemonic, traits>;

class Poptorch_Op<string mnemonic, list<Trait> traits = [], MemoryEffects memory_effect = NoSideEffect> :
        Poptorch_BasicOp<mnemonic, traits # [DeclareOpInterfaceMethods<Poptorch_PoplarImplInterface>, memory_effect]>;

class Poptorch_NotImplementedOp<string mnemonic, list<Trait> traits = [], MemoryEffects memory_effect = NoSideEffect> : Poptorch_BasicOp<mnemonic, traits # [NotImplementedOp, memory_effect]>;

class Poptorch_AbstractOp<string mnemonic, list<Trait> traits = [], MemoryEffects memory_effect = NoSideEffect> :
        Poptorch_BasicOp<mnemonic, traits # [memory_effect]>;

// Internal ops are created from poptorch dialect passes but cannot be built directly
class Poptorch_InternalOp<string mnemonic, list<Trait> traits = [], MemoryEffects memory_effect = NoSideEffect> :
        Op<Poptorch_Dialect, mnemonic, traits # [DeclareOpInterfaceMethods<Poptorch_PoplarImplInterface>, memory_effect]>;

class Poptorch_AbstractInternalOp<string mnemonic, list<Trait> traits = [], MemoryEffects memory_effect = NoSideEffect> :
        Op<Poptorch_Dialect, mnemonic, traits # [memory_effect]>;

class Poptorch_ViewOp<string mnemonic, string outplaceTypeName, string inverseTypeName, list<Trait> traits = []>
    : Poptorch_AbstractOp<mnemonic, traits # [Poptorch_ViewInterface, ViewOp]> {
  let extraClassDeclaration = [{
    using OutplaceOp = }] # outplaceTypeName # [{;
    using InverseOp = }] # inverseTypeName # [{;
  }];
}

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
def Poptorch_tensorlist_no_grad : Variadic<Poptorch_tensor_no_grad>;

#endif // POPTORCH_IR_DIALECT
