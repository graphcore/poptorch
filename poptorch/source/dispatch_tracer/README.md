## Dispatch tracing

Dispatch tracing is our own implementation of torch::jit::trace which allows us
to sidestep some of the constraints of that API as well as trace autograd functions.

We support two backends. 

- JIT : Traces the incoming user model into normal PyTorch JIT IR first then
        canonicalises them into our PopART compatible JIT IR.
- MLIR: Traces the model directly into our PyTorch native MLIR backend. Can use
        the above mechanism internally to decompose operations into the PopART
        subset or support them directly.

RegisterAtenOverloads intercepts the initial call from PyTorch then directs that
to whichever backend is active. A backend must provide a fallback operation and
a function for any overloaded PyTorch function which cannot be "boxed" or has
unique properties which make it easier.

# JIT

JIT works by using the normal PyTorch JIT API to turn the given OperatorHandle and Stack (of at::tensors/scalar/vector types) into JIT nodes. We then canonicalise that into our own IR.

Once the graph has been traced, the traced graph can be retrieved and used in our compile process as a stand in for the normal torch::jit::Trace compiledgraph. Most cleanup stages are no longer required at this point.

Models can still only be traced in inference mode, with PopART optionally applying its own autograd to turn the traced inference graph into a training graph.


# MLIR

MLIR is somewhat more complex as it is able to trace more of the graph as it uses the PyTorch autograd and gradients directly. This means it gets exposed to more of PyTorch so must handle more unexpected but legal inputs. For example in the autograd PyTorch stores variables for later processing, like the forward input to be later retrieved in the backward pass. In some of these cases PyTorch will softcopy the tensor by just swapping the storage pointer. However to our eyes it is a new tensor. So in the MLIR path we have to handle more tensor to value resolution code.

Other than having to faithfully lower more varied legal input than in JIT the main difference is that it has two paths to lower a node.

- It can use the JIT path to guarantee it can support at least as much as PopART and reuses our canonicalisation code to break down nodes further.
- It can directly map a torch operation onto IR without needing canonicalisation.

See CompilerDispatchTable.cpp for all the calls. The API with MLIR is generated automatically by MLIR and can be seen in the poptorch_compiler pytorch_bridge include folder. 

- DirectlySupportedOps.h.inc : Maps aten operations directly onto an MLIR operation.
- PopartAPISupportedOps.h.inc: Maps aten operations onto the PopART subset via unpacking JIT arguments, just like LowerToPopart.

# Code overview

| File | Description |
| ---- | --- |
| RegisterAtenOverloads.cpp | Dispatcher point of first contact. Registers hooks with PyTorch to pick up the incoming calls. |
| ValueMapper.cpp/hpp | Handles some state/logic to help map at::Tensors onto IR values and MLIR Tensors. |
| CommonHelperFunctions.cpp/hpp | Helper functions used by JIT and MLIR backends which handle the JIT graph. |
| dispatchers | Folder containing the backend specific dispatch code. |
| Tracer.hpp | Abstract backend definition. |
| JitDispatch.hpp/cpp | Contains the implementation of the JIT backend. |
| MLIRDispatch.hpp/cpp | Contains the implementation of the MLIR backend. |
| CompilerDispatchTable.cpp | Dispatch table used by MLIR backend |

See MLIR section for details on DirectlySupportedOps/PopartAPISupportedOps.
