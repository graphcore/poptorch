.. _efficient_data_batching:

=======================
Efficient data batching
=======================

By default PopTorch will process the `batch_size` which you provided to
the :py:class:`poptorch.DataLoader`.

When using the other options below, this actual number of samples used per step
varies to allow the IPU(s) to process data more efficiently.

However, the effective (mini-)batch size for operations which depend on it (such
as batch normalization) will not change. All that changes is how much data is
actually sent for a single step.

.. note:: Failure to use :py:class:`poptorch.DataLoader` may result in
   accidentally changing the effective batch size for operations which depend on
   it such as batch normalization.

poptorch.DataLoader
===================

If you set the `DataLoader` `batch_size` to more than 1 then each operation
in the model will process that number of elements at any given time.

.. autoclass:: poptorch.DataLoader
   :special-members: __init__
   :members:

poptorch.AsynchronousDataAccessor
=================================

To reduce host overhead the data loading process can be offloaded to a
separate thread using an :py:class:`~poptorch.AsynchronousDataAccessor`.
Doing this allows you to reduce the host/IPU communication overhead by
using the time that the IPU is running to load the next batch on the
CPU. This means when the IPU is finished executing and returns to host
the data will be ready for it to pull in again.

.. autoclass:: poptorch.AsynchronousDataAccessor
   :special-members: __init__
   :members: terminate

Example
-------

.. literalinclude:: device_iterations.py
  :caption: Use of AsynchronousDataAccessor
  :start-after: data_accessor_start
  :end-before: data_accessor_end
  :emphasize-lines: 10
  :linenos:

poptorch.Options.deviceIterations
=================================

If you set :py:meth:`~poptorch.Options.deviceIterations` to more
than 1 then you are telling PopART to execute that many batches in serial.

Essentially, it is the equivalent of launching the IPU in a loop over that
number of batches. This is efficient because that loop runs on the IPU
directly.

Example
-------

.. literalinclude:: device_iterations.py
  :caption: Use of device iterations and batch size
  :start-after: iterations_start
  :end-before: iterations_end
  :emphasize-lines: 55, 61
  :linenos:

poptorch.Options.replicationFactor
==================================

:py:meth:`~poptorch.Options.replicationFactor` will replicate the model over N
IPUs to allow automatic data parallelism across many IPUs.

.. literalinclude:: device_iterations.py
  :caption: Use of replication factor
  :start-after: replication_start
  :end-before: replication_end
  :emphasize-lines: 8
  :linenos:


poptorch.Options.Training.gradientAccumulation
==============================================

You need to use :py:meth:`~poptorch.options._TrainingOptions.gradientAccumulation`
when training with pipelined models because the weights are shared across
pipeline batches so gradients will be both updated and used by subsequent batches
out of order.

.. seealso:: :py:class:`poptorch.Block`

.. literalinclude:: device_iterations.py
  :caption: Use of gradient accumulation
  :start-after: gradient_acc_start
  :end-before: gradient_acc_end
  :emphasize-lines: 8
  :linenos:
