.. _supported_ops:

IPU supported operations
************************

Below is a list of currently supported operations that can be
executed on IPU hardware. This list will be expanded over time
as we add more support. Some overloads and modes of operation
for ops are not supported and we've tried to list all the caveats
but some may have been missed.


Torch operations
================

Tensor operations
-----------------

Many of the tensor operations will be executed before even reaching the IPU
so we can consider them supported anyway. Some, like ``contiguous()``, make
no sense on a distributed memory system like the IPU so are ignored. There
are no constraints on the memory format of how operations should be called
other than the constraint that all graph inputs should be contiguous.

We will also create tensor views. However, the aliasing property of views
with respect to in-place operations should not be relied on as we may have slightly different
view behaviour.

Additionally some PyTorch operations may be implemented by composition of
the listed ops but may not be explicitly listed but are in fact supported.


Creation ops
''''''''''''

* ``torch.arange``
* ``tensor.fill``
* ``torch.full``
* ``torch.full_like``
* ``torch.ones``
* ``torch.zeros``

Indexing, Slicing, Joining, Mutating Ops
''''''''''''''''''''''''''''''''''''''''

PyTorch functions

* ``torch.cat``
* ``torch.chunk``
* ``torch.reshape``
* ``torch.stack``
* ``torch.split``
* ``torch.squeeze``
* ``torch.t``
* ``torch.transpose``
* ``torch.unsqueeze``
* ``torch.where``

Tensor methods

* ``tensor.expand``
* ``tensor.expand_as``
* ``tensor.masked_fill``

Random Samplers
'''''''''''''''
To set the random state use poptorch.Options.randomSeed

* ``torch.distributions.Uniform``
* ``torch.normal``
* ``torch.rand``
* ``torch.randn``
* ``torch.uniform``

Math operations
---------------

Pointwise Ops
'''''''''''''

* ``torch.abs``
* ``torch.add``
* ``torch.asin``
* ``torch.atan``
* ``torch.ceil``
* ``torch.clamp``
* ``torch.cos``
* ``torch.cosh``
* ``torch.div``
* ``torch.exp``
* ``torch.expm1``
* ``torch.floor``
* ``torch.floor_divide``
* ``torch.frac``
* ``torch.log``
* ``torch.log10``
* ``torch.log1p``
* ``torch.log2``
* ``torch.mul``
* ``torch.norm``
* ``torch.neg``
* ``torch.pow``
* ``torch.reciprocal``
* ``torch.round``
* ``torch.rsqrt``
* ``torch.sigmoid``
* ``torch.sign``
* ``torch.sin``
* ``torch.sinh``
* ``torch.sqrt``
* ``torch.square``
* ``torch.sub``
* ``torch.tan``
* ``torch.tanh``
* ``torch.true_divide``
* ``torch.trunc``


Reduction Ops
'''''''''''''

* ``torch.argmax``
* ``torch.argmin``
* ``torch.mean``
* ``torch.prod``
* ``torch.logsumexp``
* ``torch.sum``


Comparison Ops
''''''''''''''

* ``torch.eq``
* ``torch.ge``
* ``torch.gt``
* ``torch.le``
* ``torch.lt``

    torch.min and torch.max only support (tensor, tensor) and (tensor) overloads. They do
    not support the (tensor, dim=.*, keepdim=.*) overload.

* ``torch.max``
* ``torch.min``
* ``torch.ne``
* ``torch.isnan``

    torch.topk only supports sorted=True and Largest=True arguments.

* ``torch.topk``

BLAS and LAPACK Operations
''''''''''''''''''''''''''

* ``torch.addmm``
* ``torch.matmul``
* ``torch.bmm``


Torch.nn operations
===================

Containers
----------

``torch.nn.Module`` and ``torch.nn.Sequential`` can be passed into our
compiler wrappers and just work.


Convolution layers
------------------

Conv transpose operations do not yet support dilations.

* ``torch.nn.Conv1d``
* ``torch.nn.Conv2d``
* ``torch.nn.Conv3d``
* ``torch.nn.ConvTranspose1d``
* ``torch.nn.ConvTranspose2d``
* ``torch.nn.ConvTranspose3d``


Pooling layers
--------------

Currently the max pool layers do not return the indices
so only the variants with ``return_indices=False`` are supported.

* ``torch.nn.MaxPool1d``
* ``torch.nn.MaxPool2d``
* ``torch.nn.MaxPool3d``
* ``torch.nn.AvgPool1d``
* ``torch.nn.AvgPool2d``
* ``torch.nn.AvgPool3d``
* ``torch.nn.AdaptiveAvgPool2d``

Padding layers
--------------

All padding layers are supported.

* ``torch.nn.ReflectionPad1d``
* ``torch.nn.ReflectionPad2d``
* ``torch.nn.ReplicationPad1d``
* ``torch.nn.ReplicationPad2d``
* ``torch.nn.ReplicationPad3d``
* ``torch.nn.ZeroPad2d``
* ``torch.nn.ConstantPad1d``
* ``torch.nn.ConstantPad2d``
* ``torch.nn.ConstantPad3d``


Activations
-----------

* ``torch.nn.ELU``
* ``torch.nn.GELU``
* ``torch.nn.LeakyReLU``
* ``torch.nn.LogSoftmax``
* ``torch.nn.ReLU``
* ``torch.nn.SELU``
* ``torch.nn.Sigmoid``
* ``torch.nn.Softmax``
* ``torch.nn.Softsign``
* ``torch.nn.Tanh``
* ``torch.nn.PReLU``
* ``torch.nn.Hardtanh``
* ``torch.nn.functional.glu``


Normalization layers
--------------------

Currently only ``affine=True`` is supported as a parameter. That is to say, only the variants with trainable parameters are supported.

* ``torch.nn.BatchNorm1d``
* ``torch.nn.BatchNorm2d``
* ``torch.nn.BatchNorm3d``
* ``torch.nn.LayerNorm``
* ``torch.nn.GroupNorm``

Recurrent layers
----------------

* ``torch.nn.LSTM``

Linear layers
-------------

* ``torch.nn.Identity``
* ``torch.nn.Linear``
* ``torch.nn.Bilinear``

Dropout
-------

* ``torch.nn.dropout``

Sparse layers
-------------

Embedding is supported with the exception of ``padding_idx`` being ignored.

* ``torch.nn.Embedding``

Loss functions
--------------

This version supports a limited subset of loss functions. However, we support
:py:func:`poptorch.identity_loss` which gives you the ability to implement any arbitrary
loss function.

.. seealso:: :py:func:`poptorch.identity_loss`

One caveat for the following loss functions is if they are used they will always be included
in the back propagation and will always receive a gradient, which is a slight deviation from
normal PyTorch operations, where they have to opt in to the gradient pass.

* ``torch.nn.L1Loss``
* ``torch.nn.MSELoss``
* ``torch.nn.CrossEntropyLoss``
* ``torch.nn.NLLLoss``
* ``torch.nn.BCELoss``

Vision Layers
-------------
Only nearest is supported.

* ``torch.nn.Upsample``


.. _float_16_op_support:

Float 16 operations
===================

Due to the limitation of PyTorch's float 16 support on the CPU (used for tracing the model), certain operations may result in the use of float 32 where float 16 would be expected, or float 16 where float 32 would be expected.
This is because the model must always be traced with float 16 inputs converted to float 32.

Casting
-------
The ``tensor.to(dtype)`` argument will be ignored because it may refer to one or more float 16 tensors which were converted to float 32 to allow tracing to happen, for example ``a.to(b.dtype)`` where ``b`` may be a float 16 tensor converted to a float 32 tensor.
Once the output of the op or one of its descendants encounters a known float 16 or float 32 input, the type will be resolved to this type.

The following examples show cases where the casting functionality is resolved based on context, correctly or incorrect:

.. literalinclude:: half_float_casting.py
    :language: python
    :caption: Cases where casting resolves to the correct type
    :linenos:
    :lines: 8-34

.. literalinclude:: half_float_casting.py
    :language: python
    :caption: Cases where casting resolves to an incorrect type
    :linenos:
    :lines: 38-58


Creation functions
------------------

The following functions are affected:
* torch.ones
* torch.rand
* torch.zeros
* torch.distributions.uniform.Uniform

The ``dtype`` arguments will be ignored because they may refer to  float 16 tensors which were converted to float 32 tensors to allow tracing to succeed.
Once the output of the op, or its descendant, encounters a known float 16 or float 32 input, the ``dtypes`` are resolved to this type.

The following examples show cases where the type output differs from PyTorch:

.. literalinclude:: half_float_ops.py
    :language: python
    :caption: Type resolution when using torch.zeros
    :linenos:
    :lines: 8-28


.. literalinclude:: half_float_ops.py
    :language: python
    :caption: Type resolution when using torch.rand
    :linenos:
    :lines: 32-52


.. literalinclude:: half_float_ops.py
    :language: python
    :caption: Type resolution when using torch.distributions.uniform.Uniform
    :linenos:
    :lines: 56-79
