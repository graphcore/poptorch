=====================
Experimental features
=====================

Distributed execution without PopRun
====================================

PopTorch supports distributed execution on a Pod using the IPU over Fabric
(IPUoF).

If you run a program using your own distributed processing tool instead of PopRun, the only change you need to make to your code is to set the ID of the current process and
the total number of processes the execution is distributed across, using
:py:meth:`~poptorch.options._DistributedOptions.configureProcessId`.

Note that :py:meth:`~poptorch.Options.replicationFactor` should
be used to set the number of local replicas (per host) not the total (global)
number of replicas.

.. literalinclude:: device_iterations.py
  :caption: Changes required for distributed execution
  :start-after: distributed_execution_start
  :end-before: distributed_execution_end
  :emphasize-lines: 9, 12, 18
  :linenos:

.. note:: The ``DataLoader`` will automatically select a different subset of the
  dataset based on the process ID.

.. warning:: All the processes must use the same seed if ``shuffle=True`` is used
  for the ``DataLoader``.

torch.nn.CTCLoss
================

The CTCLoss operator is supported, with some limitations:

#. The ``reduction`` parameter must be set to either ``sum`` or ``mean``
#. The ``targets`` tensor must be 2D, corresponding to stacked, padded layout

.. _dispatcher-support:

Dispatcher support
==================

By default PopTorch uses `torch.jit.trace <https://pytorch.org/docs/1.10.0/generated/torch.jit.trace.html#torch.jit.trace>`_ to build a static graph representation of a torch.nn.Module.

However, this approach suffers from several limitations:
* Only tensors can be passed as arguments.
* The traced model will run on the CPU as part of the tracing process.

  * This is expensive for large batch sizes.
  * This means we need to add workarounds to trace types which are not supported on the CPU, for example FP16 (See ref:`float_16_op_support` for more details).

* Source code location is not supported: most of the instructions will point at torch.nn.module.py rather than user code.

To address these issues we have started using `the PyTorch dispatcher <http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/>`_ to build the PopTorch graph ourselves.

.. warning:: The support for the dispatcher is currently experimental: there are several known issues and we are actively working on fixing them, but the dispatcher will eventually become the default frontend in PopTorch.

To try the dispatcher see :py:meth:`~poptorch.options._JitOptions.traceModel`.

Note: while any argument type can be used in the forward method with the dispatcher, only tensors can change between model invocations as other types will be statically compiled inside the executable.
