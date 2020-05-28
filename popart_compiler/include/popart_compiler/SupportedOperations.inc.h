/*
    OP_DECLS are in the following form:
    OP_DECL(string, function, onnx implementation, arguments, body argument)
     - string is the string representation of the function
     - function is the actual op part of the <namespace>:<op> pair and will be
   used to name/call the given function.
     - Onnx implementation is the underlaying onnx function which will be
   called.
     - Arguments are the arguments to the op which will be parsed by different
   macros depending on which file this is in.
     - Body arguments are just the names of the arguments so they can be used in
   the cpp file.
     - Indexing (TODO REMOVE) this is to allow us to index into the output. We
   will have to change this in the future.
*/
#include "CompilerOperationMacros.inc"
#include "ManuallyAddedOperations.inc.h"