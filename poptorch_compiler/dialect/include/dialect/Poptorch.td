// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#ifndef POPTORCH_OPS
#define POPTORCH_OPS

include "PoptorchDialect.td"

// Now included in activations.
// include "ops/ElementWise.td"

include "ops/Activations.td"
include "ops/Linalg.td"
include "ops/Losses.td"
include "ops/Random.td"
include "ops/Reductions.td"
include "ops/NormOps.td"
include "ops/PoolingOps.td"
include "ops/Reshapes.td"
include "ops/RNN.td"
include "ops/TensorOps.td"

include "ops/Poptorch.td"

include "mlir/IR/SymbolInterfaces.td"

/*
class Poptorch_Stream<name> : Poptorch_InternalOp<name, [Symbol]> {
  let arguments = (ins SymbolAttr:$sym_name, TypeAttr:$type);

  let assemblyFormat = [{ $sym_name `:` $type attr-dict }];
}

def Poptorch_ReadStream : Poptorch_Stream<"read_stream"> {
  let summary = "Defines a stream for reading from the host.";
}

def Poptorch_WriteStream : Poptorch_Stream<"write_stream"> {
  let summary = "Defines a stream for writing back to the host.";
}
*/

/*
 * In out stream copies.
 *
 * Note these aren't registered as Poptorch_BasicOp-s because they don't need to be registered in the builder
 */
def Poptorch_copy_from_host
    : Op<Poptorch_Dialect, "copy_from_host", [DeclareOpInterfaceMethods<Poptorch_PoplarImplInterface>]> {
// TODO(T69660): support pruning of copy_from_host properly
//    : Poptorch_InternalOp<"copy_from_host", [], MemoryEffects<[MemRead]>> {
  let arguments = (ins StrAttr:$handle);
  let results = (outs Poptorch_tensor:$result);
  let assemblyFormat = [{`(` $handle `)` `->` `(`type($result)`)` attr-dict }];
  let summary = "Copy tensor from host.";
}

def Poptorch_copy_to_host: Poptorch_InternalOp<"copy_to_host", [], MemoryEffects<[MemWrite]>> {
  let arguments = (ins Poptorch_tensor:$tensor, StrAttr:$handle);
  let assemblyFormat = [{`(` $tensor `,` $handle`)` `(`type($tensor)`)` attr-dict }];
  let summary = "Copy tensor to host.";
}

def Poptorch_global_tensor_op : Op<Poptorch_Dialect, "global", [Symbol]> {
  let summary = "Defines a global tensor varibale.";

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);

  let assemblyFormat = [{ $sym_name `:` $type attr-dict }];
}

def Poptorch_copy_from_global_state
    : Poptorch_InternalOp<"copy_from_global_state", [DeclareOpInterfaceMethods<SymbolUserOpInterface>], MemoryEffects<[MemRead]>> {
  let arguments = (ins FlatSymbolRefAttr:$handle);
  let results = (outs Poptorch_tensor:$tensor);
  let assemblyFormat = [{`(` $handle `)` `->` `(`type($tensor)`)` attr-dict }];
  let summary = "Mark the tensor as being copied into the funciton from the global store";
  let description = [{
    The global store is for tensor that are used in other progarms in the poplar
    graph. This is useful for marking parameters since these will be copied to
    and from the host in a separate program.
  }];
}

def Poptorch_copy_to_global_state
    : Poptorch_InternalOp<"copy_to_global_state", [DeclareOpInterfaceMethods<SymbolUserOpInterface>], MemoryEffects<[MemWrite]>> {
  let arguments = (ins Poptorch_tensor:$tensor, FlatSymbolRefAttr:$handle);
  let assemblyFormat = [{`(` $tensor `,` $handle`)` `(`type($tensor)`)` attr-dict }];
  let summary = "Mark the tensor as being written from the function into the global store";
  let description = [{
    The global store is for tensor that are used in other progarms in the poplar
    graph. This is useful for marking parameters since these will be copied to
    and from the host in a separate program.
  }];
}


def Poptorch_output_tensor: Poptorch_AbstractOp<"output_tensor", [], MemoryEffects<[MemWrite]>> {
    let arguments = (ins Poptorch_tensor:$tensor, I64Attr:$tensorId);
    let assemblyFormat = [{`(` $tensor `,` $tensorId`)` `(`type($tensor)`)` attr-dict }];
    let summary = "Mark a tensor as being an output.";
}

#endif // POPTORCH_OPS
