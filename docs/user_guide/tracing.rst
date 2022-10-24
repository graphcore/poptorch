=======================
Legacy tracing frontend
=======================

.. warning::
    Tracing has been deprecated since PopTorch 3.0. We suggest you use the
    dispatcher frontend, which is enabled by default and brings many benefits,
    including greatly simplified handling of ``float16`` operations. However,
    if you need to use tracing for legacy reasons, this section explains the
    limitations imposed and the workarounds available.

.. _dispatcher-support:

Dispatcher support
==================

Up to version 2.6  PopTorch used `torch.jit.trace <https://pytorch.org/docs/1.10.0/generated/torch.jit.trace.html#torch.jit.trace>`_ to build a static graph representation of a torch.nn.Module.

However, this approach suffered from several limitations:

* Only tensors could be passed as arguments.
* The traced model ran on the CPU as part of the tracing process.

  * It was expensive for large batch sizes.
  * It meant we needed to add workarounds to trace types which were not supported on the CPU, for example ``float16`` (See :ref:`tracing-float16` for more details).

* Source code location was not supported: most of the instructions pointed at ``torch.nn.module.py`` rather than at user code.

To address these issues the default is now to use the `PyTorch dispatcher <http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/>`_ to build the PopTorch graph ourselves.

.. _tracing-constraints:

Constraints when using tracing
==============================

When tracing, PopTorch uses PyTorch's `torch.jit.trace <https://pytorch.org/docs/1.10.0/generated/torch.jit.trace.html#torch.jit.trace>`_
API. This means that the tracing frontend inherits the constraints of that API. These include:

   * Inputs must be PyTorch tensors or tuples containing PyTorch tensors.
   * ``None`` can be used as a default value for a parameter but cannot be
     explicitly passed as an input value.

See also :numref:`constraints` for general PopTorch constraints.

.. _tracing-float16:

16-bit float operations when using tracing
==========================================

.. note::
    Handling of ``float16`` operations is greatly simplified with the dispatcher frontend. For help on migrating
    ``float16`` code from tracing to the dispatcher frontend, see :ref:`float_16_migration`.

Due to the limitation of PyTorch's ``float16`` support on the CPU (used for tracing the model), certain operations may result in the use of ``float32`` where ``float16`` would be expected, or ``float16`` where ``float32`` would be expected.
This is because the model must always be traced with ``float16`` inputs converted to ``float32``.

This limitation is much less noticeable when ``opts.Precision.halfFloatCasting(poptorch.HalfFloatCastingBehavior.HalfUpcastToFloat)`` has not been set because PopTorch's default casting functionality is to output a ``float16`` if any input of the op is ``float16``.
In such situations, any data type which incorrectly resolves to a ``float16`` would have been cast to a ``float16`` in any case.

Casting
-------

The ``dtype`` argument in ``tensor.to(dtype)`` will be ignored if it is ``torch.float32`` because it may refer to one or more ``float16`` tensors which were converted to ``float32`` to allow tracing to happen, for example ``a.to(b.dtype)`` where ``b`` may be a ``float16`` tensor converted to a ``float32`` tensor.
Once the output of the op or one of its descendants encounters a known ``float16`` or ``float32`` input, the type will be resolved to this type.

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

* ``torch.ones``
* ``torch.rand``
* ``torch.zeros``
* ``torch.distributions.uniform.Uniform``

The ``dtype`` arguments will be ignored because they may refer to  ``float16`` tensors which were converted to ``float32`` tensors to allow tracing to succeed.
Once the output of the op, or its descendant, encounters a known ``float16`` or ``float32`` input, the ``dtype`` values are resolved to this type.

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

Some normalization layers require the computation of running statistics - mean and variance. These tensors will be computed as ``float32`` even though the inputs to the operator can be ``float16``. This behaviour has been chosen to strike a balance between performance and numerical accuracy.

The following operators are affected:

* ``torch.nn.BatchNorm1d``
* ``torch.nn.BatchNorm2d``
* ``torch.nn.BatchNorm3d``

The type of running statistics computations may be controlled via ``opts.Precision.runningStatisticsAlwaysFloat(bool)``. For example, in the script below, mean and variance computations will be performed in half-precision:

.. literalinclude:: running_statistics_half.py
    :language: python
    :caption: Controlling type of running mean and variance computations
    :linenos:
    :start-after: half_stats_begin
    :end-before: half_stats_end

.. _tracing-autocast:

Automatic mixed-precision casting
=================================

.. warning::
    The autocasting API is only available when using the legacy tracing frontend. When using the dispatcher frontend,
    which is the default frontend, simply use PyTorch casting.

PopTorch supports converting your model automatically between ``float16`` and ``float32``.
This functionality is not active by default - you must enable it explicitly by
calling the ``autocast(enabled=True)`` method at model level.

.. literalinclude:: autocast.py
    :language: python
    :caption: Enabling automatic casting at model level
    :start-after: model_autocast_start
    :end-before: model_autocast_end
    :emphasize-lines: 2

During compilation, selected layers and operators will have their types adjusted aiming
to strike a good compromise between compute efficiency, memory requirements and numerical
precision.

You can also set automatic casting at the layer level. In this situation,
its effect is hierarchical: changing the setting for a layer affects it and all layers it contains.

In the following example, automatic casting is enabled for all layers of the model, except for the
first activation and second convolution.

.. literalinclude:: autocast.py
    :language: python
    :caption: Controlling automatic casting at layer level
    :start-after: layer_autocast_start
    :end-before: layer_autocast_end
    :emphasize-lines: 6-8

You can also set automatic casting with the function decorator ``@poptorch.autocast(enabled=True)``.
Its effect is to apply automatic casting to the body of the function. Setting its parameter to ``False``
has the opposite effect. A typical use-case is applying it to the ``forward`` function of custom modules.

.. literalinclude:: autocast.py
    :language: python
    :caption: Controlling automatic casting via decorator
    :start-after: decorator_autocast_start
    :end-before: decorator_autocast_end
    :emphasize-lines: 2

In addition, you can apply ``poptorch.autocast(enabled=True)`` to a code-block, with similar effect.

.. literalinclude:: autocast.py
    :language: python
    :caption: Applying automatic casting to a code-block
    :start-after: block_autocast_start
    :end-before: block_autocast_end
    :emphasize-lines: 3

You can disable this feature for the whole application via the ``autocastEnabled(bool)``
method of :py:class:`~poptorch.options._PrecisionOptions`.

.. literalinclude:: autocast.py
    :language: python
    :caption: Disabling automatic casting
    :start-after: disable_autocast_start
    :end-before: disable_autocast_end
    :emphasize-lines: 2

Custom casting policies
-----------------------

PopTorch provides a mechanism to customize automatic casting behaviour in the form of casting
policy classes. A casting policy is defined by four sets of PyTorch modules and/or PyTorch operators:

#. ``fp16`` - set of operations to be typed as ``float16``
#. ``fp32`` - set of operations to be typed as ``float32``
#. ``promote`` - set of operations to be promoted to ``float32`` should they take mixed-precision inputs
#. ``demote`` - set of operations to be demoted to ``float16`` should they take mixed-precision inputs

The following example describes a policy where convolution and ReLU operations are to be performed using ``float16``,
whilst batch matrix multiplication is to be performed using ``float32``. Dot product computations will be
promoted to ``float32`` when operands have mixed precision.

.. literalinclude:: autocast.py
    :language: python
    :caption: Custom casting policies
    :start-after: policy_autocast_start
    :end-before: policy_autocast_end
    :emphasize-lines: 5, 8
