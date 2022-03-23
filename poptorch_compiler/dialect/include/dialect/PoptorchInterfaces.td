// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_INTERFACES_TD_
#define POPTORCH_INTERFACES_TD_

include "mlir/IR/OpBase.td"

def Poptorch_PoplarImplInterface : OpInterface<"PoplarImplInterface"> {
  let description =
      [{Any operation with this interface can be lowered to
            poplar.The definition must be linked into lower_to_poplar.}];

  let methods = [
    InterfaceMethod<[{Lowers this operation to a poplar implementation.}],
                    "void", "lowerToPoplar",
                    (ins "poptorch_ir::CompilerContext&"
                     : $context)>,
  ];
}

#endif // POPTORCH_INTERFACES_TD_
