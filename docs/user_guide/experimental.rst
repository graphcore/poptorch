=====================
Experimental features
=====================

Distributed execution without PopRun
====================================

PopTorch supports distributed execution on IPU-POD using the IPU over Fabric
(IPUoF).

If you run using your own distributed processing tool instead of PopRun, the only change to your code needed is to set the id of the current process and
the total number of processes the execution is distributed across using
:py:meth:`~poptorch.options._DistributedOptions.configureProcessId`.
Please also be aware that :py:meth:`~poptorch.Options.replicationFactor` should
be used to set the number of local replicas (per host) not the total (global)
number of replicas.

.. literalinclude:: device_iterations.py
  :caption: Changes required for distributed execution
  :start-after: distributed_execution_start
  :end-before: distributed_execution_end
  :emphasize-lines: 9, 12, 18
  :linenos:

.. note:: The ``DataLoader`` will automatically select a different subset of the
  dataset based on the process id.

.. warning:: All the processes must use the same seed if ``shuffle=True`` is used
  for the ``DataLoader``.

torch.nn.CTCLoss
================

Support was added for the CTCLoss operator with a number of limitations:
#. ``zero_infinity`` parameter must be set ``False``
#. ``reduction`` parameter must be set to either ``sum`` or ``mean``
#. ``targets`` tensor must be 2D, corresponding to stacked, padded layout
