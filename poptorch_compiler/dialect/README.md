# Dialect

The include folders contain most of the tablegen and the PopTorchDialect file contains the only manually written C++ code required to back the tablegen.


# PopTorch IR Features

- Inplace operations are directly supported and suffixed with a `_`. In the future we will likely add an inplace annotation on the node.

- The class `Poptorch_Op` defines an operation as having a poplar implementation.
    - This works by applying the `PoplarImplInterface` behind the scenes.

- Operations which should exist in the IR but don't require an implementation should inherit from `Poptorch_AbstractOp`

- The type `Poptorch_tensor` represents a tensor
    - TODO(T49567): We will provide some kind of type mechanism to represent parameters, buffers, and gradients directly.

- The trait `ViewOp` on an operation marks it as being a view change operation on the tensor.


# Writing an IR node.

If the class does not return anything then you only need to define the inputs. If it does return something then you either need to inherit from a trait which establishes a rule for deducing the output type or a builder which deduces the output type. To work with out autogeneration script, the API must provide a builder which takes in the C++ equivalent of its inputs.