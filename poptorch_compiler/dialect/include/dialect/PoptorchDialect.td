
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

class Poptorch_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Poptorch_Dialect, mnemonic, traits # [DeclareOpInterfaceMethods<Poptorch_PoplarImplInterface>]>;


class Poptorch_AbstractOp<string mnemonic, list<OpTrait> traits = []> :
        Op<Poptorch_Dialect, mnemonic, traits>;


def valid_ints : SignedIntOfWidths<[8, 16, 32]>;
def valid_uints : UnsignedIntOfWidths<[8, 16, 32]>;
def valid_float : FloatOfWidths<[16, 32]>;

def valid_tensor : StaticShapeTensorOf<[valid_ints, valid_uints, valid_float]>;

/*
    We will add in specific types for parameters and buffers later.
*/
// Any valid tensor type.
def Poptorch_tensor : AnyTypeOf<[valid_tensor]>;

def Poptorch_tensorlist : Variadic<Poptorch_tensor>;

#endif // POPTORCH_IR_DIALECT
