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

* ``torch.zeros``
* ``torch.ones``
* ``torch.arange``


Indexing, Slicing, Joining, Mutating Ops
''''''''''''''''''''''''''''''''''''''''

* ``torch.cat``
* ``torch.chunk``
* ``torch.reshape``
* ``torch.split``
* ``torch.squeeze``
* ``torch.t``
* ``torch.transpose``
* ``torch.unsqueeze``
* ``tensor.expand``
* ``tensor.expand_as``


Math operations
---------------

Pointwise Ops
'''''''''''''

* ``torch.abs``
* ``torch.add``
* ``torch.asin``
* ``torch.atan``
* ``torch.ceil``
* ``torch.cos``
* ``torch.cosh``
* ``torch.div``
* ``torch.exp``
* ``torch.expm1``
* ``torch.floor``
* ``torch.frac``
* ``torch.log``
* ``torch.log10``
* ``torch.log1p``
* ``torch.log2``
* ``torch.mul``
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


BLAS and LAPACK Operations
''''''''''''''''''''''''''

* ``torch.addmm``
* ``torch.matmul``


Torch.nn operations
===================

Containers
----------

``torch.nn.Module`` and ``torch.nn.Sequential`` can be passed into our
compiler wrappers and just work.


Convolution layers
------------------

* ``torch.nn.Conv2d``


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

LSTM only supports the default options for parameters ``batch_first`` (False), ``dropout`` (off/0), and ``bias`` (True).

* ``torch.nn.LSTM``

Linear layers
-------------

* ``torch.nn.Identity``
* ``torch.nn.Linear``

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
``poptorch.identity_loss`` which gives users the ability to implement any arbitrary
loss function. See operation explanation in the overview.

One caveat for the following loss functions is if they are used they will always be included
in the back propagation and will always receive a gradient, which is a slight deviation from
normal PyTorch operations, where they have to opt in to the gradient pass.

* ``torch.nn.L1Loss``
* ``torch.nn.MSELoss``
* ``torch.nn.CrossEntropyLoss``
* ``torch.nn.NLLLoss``
* ``torch.nn.BCELoss``
