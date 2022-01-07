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
with respect to in-place operations should not be relied on as we may have slightly different view behaviour.

Additionally, some PyTorch operations may be implemented by composition of
the listed ops but may not be explicitly listed but are in fact supported.


Creation ops
''''''''''''

* ``torch.arange``
* ``tensor.fill``
* ``torch.full``
* ``torch.full_like``
* ``torch.Tensor.new_ones``
* ``torch.Tensor.new_zeros``
* ``torch.ones``
* ``torch.ones_like``
* ``torch.zeros``
* ``torch.zeros_like``

Indexing, slicing, joining and mutating ops
'''''''''''''''''''''''''''''''''''''''''''

In PyTorch, slicing a tensor is accessing a subset of the tensor by providing the start and end indices, such as ``tensor[1:5]``.

With a PopTorch model, you may take a slice of a tensor only if one of two conditions are met:
* The start and end are constants, or can be resolved to be constants (for example, a function of the shape of a tensor which does not change between runs).
* The start and end of the slice are related by a constant, for example ``tensor[x:x+5]``. Please note that this will produce different results to PyTorch if the end value exceeds the length of the tensor: PyTorch will output a smaller size tensor but PopTorch will allow the slice to wrap round to the start of the relevant dimension.

PyTorch functions

* ``torch.cat``
* ``torch.chunk``
* ``torch.gather``
* ``torch.index_select``
* ``torch.reshape``
* ``torch.roll``
* ``torch.scatter``
* ``torch.scatter_add``
* ``torch.stack``
* ``torch.split``
* ``torch.squeeze``
* ``torch.t``
* ``torch.transpose``
* ``torch.unbind``
* ``torch.unsqueeze``
* ``torch.where``

Tensor methods

* ``tensor.expand``
* ``tensor.expand_as``
* ``tensor.masked_fill``

Random samplers
'''''''''''''''
To set the random state, use ``poptorch.Options.randomSeed``

* ``torch.bernoulli``
* ``torch.distributions.Bernoulli``
* ``torch.randn``
* ``torch.normal``
* ``torch.distributions.Normal``
* ``torch.rand``
* ``torch.uniform``
* ``torch.distributions.Uniform``

Math operations
---------------

Pointwise ops
'''''''''''''

* ``torch.abs``
* ``torch.acos``
* ``torch.acosh``
* ``torch.add``
* ``torch.addcdiv``
* ``torch.amax``
* ``torch.amin``
* ``torch.asin``
* ``torch.asinh``
* ``torch.atan``
* ``torch.atanh``
* ``torch.bitwise_and``
* ``torch.bitwise_not``
* ``torch.bitwise_or``
* ``torch.bitwise_xor``
* ``torch.ceil``
* ``torch.clamp``
* ``torch.clamp_max``
* ``torch.clamp_min``
* ``torch.cos``
* ``torch.cosh``
* ``torch.div``
* ``torch.exp``
* ``torch.expm1``
* ``torch.floor``
* ``torch.floor_divide``
* ``torch.fmod``
* ``torch.frac``
* ``torch.log``
* ``torch.log10``
* ``torch.log1p``
* ``torch.log2``
* ``torch.logical_and``
* ``torch.logical_or``
* ``torch.mul``
* ``torch.norm``
* ``torch.neg``
* ``torch.pow``
* ``torch.reciprocal``
* ``torch.remainder``
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


Reduction ops
'''''''''''''

* ``torch.argmax``
* ``torch.argmin``
* ``torch.mean``
* ``torch.median``
* ``torch.prod``
* ``torch.logsumexp``
* ``torch.std``
* ``torch.std_mean``
* ``torch.sum``
* ``torch.var``
* ``torch.var_mean``


Comparison ops
''''''''''''''

* ``torch.eq``
* ``torch.ge``
* ``torch.gt``
* ``torch.le``
* ``torch.lt``
* ``torch.max``
* ``torch.min``
* ``torch.ne``
* ``torch.isnan``

    ``torch.topk`` only supports ``sorted=True`` and ``largest=True`` arguments.

* ``torch.topk``


Other ops
'''''''''

* ``torch.cumsum``
* ``torch.cross``
* ``torch.meshgrid``
* ``torch.cartesian_prod``
* ``torch.tensordot``


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
* ``torch.nn.AdaptiveAvgPool1d``
* ``torch.nn.AdaptiveAvgPool2d``
* ``torch.nn.AdaptiveAvgPool3d``

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
* ``torch.nn.CELU``
* ``torch.nn.GELU``
* ``torch.nn.Hardshrink``
* ``torch.nn.LeakyReLU``
* ``torch.nn.LogSoftmax``
* ``torch.nn.ReLU``
* ``torch.nn.SELU``
* ``torch.nn.SiLU``
* ``torch.nn.Sigmoid``
* ``torch.nn.Softmax``
* ``torch.nn.Softplus``
* ``torch.nn.Softsign``
* ``torch.nn.Softshrink``
* ``torch.nn.Tanh``
* ``torch.nn.PReLU``
* ``torch.nn.RReLU``
* ``torch.nn.Hardtanh``
* ``torch.nn.functional.glu``
* ``torch.nn.Threshold``


Normalization layers
--------------------

Currently only ``affine=True`` is supported as a parameter. That is to say, only the variants with trainable parameters are supported.

* ``torch.nn.BatchNorm1d``
* ``torch.nn.BatchNorm2d``
* ``torch.nn.BatchNorm3d``
* ``torch.nn.LayerNorm``
* ``torch.nn.GroupNorm``
* ``torch.nn.InstanceNorm1d``
* ``torch.nn.InstanceNorm2d``
* ``torch.nn.InstanceNorm3d``

Recurrent layers
----------------

* ``torch.nn.GRU``
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

Embedding and EmbeddingBag are supported with the exception of the ``padding_idx`` parameter
being unsupported.

* ``torch.nn.Embedding``
* ``torch.nn.EmbeddingBag``
* ``torch.nn.functional.one_hot``

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
* ``torch.nn.KLDivLoss``
* ``torch.nn.PoissonNLLLoss``
* ``torch.nn.HingeEmbeddingLoss``
* ``torch.nn.BCEWithLogitsLoss``
* ``torch.nn.SmoothL1Loss``
* ``torch.nn.SoftMarginLoss``
* ``torch.nn.CosineEmbeddingLoss``
* ``torch.nn.MarginRankingLoss``
* ``torch.nn.TripletMarginLoss``
* ``torch.nn.CTCLoss``

Vision Layers
-------------
Only nearest is supported.

* ``torch.nn.Upsample``


.. _float_16_op_support:

Float 16 operations
===================

Due to the limitation of PyTorch's float 16 support on the CPU (used for tracing the model), certain operations may result in the use of float 32 where float 16 would be expected, or float 16 where float 32 would be expected.
This is because the model must always be traced with float 16 inputs converted to float 32.

This limitation is much less noticeable when ``opts.Precision.halfFloatCasting(poptorch.HalfFloatCastingBehavior.HalfUpcastToFloat)`` has not been set because PopTorch's default casting functionality is to output a float 16 if any input of the op is float 16.
In such situations, any dtype which incorrectly resolves to a float 16 would have been cast to a float 16 in any case.

Casting
-------
The ``tensor.to(dtype)`` argument will be ignored if it is ``torch.float32`` because it may refer to one or more float 16 tensors which were converted to float 32 to allow tracing to happen, for example ``a.to(b.dtype)`` where ``b`` may be a float 16 tensor converted to a float 32 tensor.
Once the output of the op or one of its descendants encounters a known float 16 or float 32 input, the type will be resolved to this type.

The following examples show cases where the casting functionality is resolved based on context, correctly or incorrectly:

.. literalinclude:: half_float_casting.py
    :language: python
    :caption: Cases where casting resolves to the correct type
    :linenos:
    :start-after: correct_cast_start
    :end-before: correct_cast_end

.. literalinclude:: half_float_casting.py
    :language: python
    :caption: Cases where casting resolves to an incorrect type
    :linenos:
    :start-after: incorrect_cast_start
    :end-before: incorrect_cast_end


Creation functions
------------------

The following functions are affected:
* torch.ones
* torch.rand
* torch.zeros
* torch.distributions.uniform.Uniform

The ``dtype`` arguments will be ignored because they may refer to  float 16 tensors which were converted to float 32 tensors to allow tracing to succeed.
Once the output of the op, or its descendant, encounters a known float 16 or float 32 input, the ``dtype`` values are resolved to this type.

The following examples show cases where the type output differs from PyTorch:

.. literalinclude:: half_float_ops.py
    :language: python
    :caption: Type resolution when using torch.zeros
    :linenos:
    :start-after: zero_res_start
    :end-before: zero_res_end


.. literalinclude:: half_float_ops.py
    :language: python
    :caption: Type resolution when using torch.rand
    :linenos:
    :start-after: rand_res_start
    :end-before: rand_res_end


.. literalinclude:: half_float_ops.py
    :language: python
    :caption: Type resolution when using torch.distributions.uniform.Uniform
    :linenos:
    :start-after: uniform_res_start
    :end-before: uniform_res_end

Normalization
-------------

Some normalization layers require the computation of running statistics - mean and variance. These tensors will be computed as float32 even though the inputs to the operator can be float16. This behaviour has been chosen to strike a balance between performance and numerical accuracy.

The following operators are affected:
* ``torch.nn.BatchNorm1d``
* ``torch.nn.BatchNorm2d``
* ``torch.nn.BatchNorm3d``

The type of running statistics computations may be controlled via ``opts.Precision.runningStatisticsAlwaysFloat(bool)``. For example, in the script below, mean and variance computations will be performed in half precision:

.. literalinclude:: running_statistics_half.py
    :language: python
    :caption: Controlling type of running mean and variance computations
    :linenos:
    :start-after: half_stats_begin
    :end-before: half_stats_end

Gradient computation control
----------------------------

``torch.no_grad`` is supported as a context manager as well as a decorator to suppress locally the
computation of the gradients.
