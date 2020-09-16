.. _efficient_data_batching:

=======================
Efficient data batching
=======================

By default PopTorch will process the `batch_size` which you provided to
the :py:class:`poptorch.DataLoader`.

When using the other options below, this changes slightly to allow PopART to
load data more efficiently.

However, the batch size actually being used on the device will not change, 
just how much data is available for PopART to load from.

poptorch.DataLoader
===================

If you set the `DataLoader` `batch_size` to more than 1 then each operation
in the model will process that number of elements at any given time.

.. autoclass:: poptorch.DataLoader
   :special-members: __init__
   :members:

poptorch.AsynchronousDataAccessor
=================================

For optimum performance the data loading process can be offloaded to a
separate thread using an :py:class:`~poptorch.AsynchronousDataAccessor`

.. autoclass:: poptorch.AsynchronousDataAccessor
   :special-members: __init__

Example
-------

.. literalinclude:: device_iterations.py
  :caption: Use of AsynchronousDataAccessor
  :lines: 146-160
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
  :lines: 3-78
  :emphasize-lines: 55, 61
  :linenos:

poptorch.Options.replicationFactor
==================================

:py:meth:`~poptorch.Options.replicationFactor` will replicate the model over N
IPUs to allow automatic data parallelism across many IPUs.

.. literalinclude:: device_iterations.py
  :caption: Use of replication factor
  :lines: 82-108
  :emphasize-lines: 8
  :linenos:


poptorch.Options.Traning.gradientAccumulation
=============================================

You need to use :py:meth:`~poptorch.options._TrainingOptions.gradientAccumulation`
when training with pipelined models because the weights are shared across
pipeline batches so gradients will be both updated and used by subsequent batches
out of order.

.. seealso:: :py:class:`poptorch.IPU`

.. literalinclude:: device_iterations.py
  :caption: Use of gradient accumulation
  :lines: 110-135
  :emphasize-lines: 8
  :linenos:
