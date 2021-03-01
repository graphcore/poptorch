=====================
Experimental features
=====================

.. _distributed_execution:

Distributed execution
=====================

PopTorch supports distributed execution on IPU-POD using the IPU over Fabric(IPUoF).
Please refer to the popdist documentation for examples.

If you run without using ``poprun``, the only change to your code needed is to set the id of the current process and
the total number of processes the execution is distributed across using
:py:meth:`~poptorch.options._DistributedOptions.configureProcessId`

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
