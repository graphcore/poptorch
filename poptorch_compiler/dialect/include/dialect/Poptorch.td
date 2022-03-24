// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPTORCH_OPS
#define POPTORCH_OPS

include "PoptorchDialect.td"

// Now included in activations.
// include "ops/ElementWise.td"

include "ops/Activations.td"
include "ops/Generated.td"
include "ops/Linalg.td"
include "ops/Losses.td"
include "ops/Random.td"
include "ops/Reductions.td"
include "ops/NormOps.td"
include "ops/PoolingOps.td"
include "ops/Reshapes.td"
include "ops/TensorOps.td"

def Poptorch_end_graph: Poptorch_AbstractOp<"end_graph", [HasParent<"::mlir::FuncOp">, Terminator]> {
    let summary = "Return op";
}

/*
 * In out stream copies.
 */
class Poptorch_StreamCopy<string name> :  Poptorch_Op<name, []> {
    let arguments = (ins Poptorch_tensor:$tensor, StrAttr:$handle);
    let assemblyFormat = [{`(` $tensor `,` $handle`)` `(`type($tensor)`)` attr-dict }];
}

def Poptorch_copy_from_host: Poptorch_StreamCopy<"copy_from_host"> {
    let summary = "Copy tensor from host.";
}

def Poptorch_copy_to_host: Poptorch_StreamCopy<"copy_to_host"> {
    let summary = "Copy tensor to host.";
}

#endif // POPTORCH_OPS




