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

.. note:: ``DataLoader`` will automatically select a different subset of the
  dataset based on the process ID.

.. warning:: All the processes must use the same seed if ``shuffle=True`` is used
  for the ``DataLoader``.

torch.nn.CTCLoss
================

The CTCLoss operator is supported, with some limitations:

#. The ``reduction`` parameter must be set to either ``sum`` or ``mean``
#. The ``targets`` tensor must be 2D, corresponding to stacked, padded layout
