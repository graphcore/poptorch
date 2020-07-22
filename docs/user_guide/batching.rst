.. _efficient_data_batching:

Efficient data batching
=======================

By default PopTorch will execute the batch size which you provided. When
using the options below, this changes slightly to allow PopART to load data more
efficiently. At all times the batch size actually being executed at any given point
on the device will not change, just how much data is available for PopART load from.

* Device iterations. If ``device_iterations`` is set to more than 1 then you are telling PopART to execute
  that many batches in serial. Essentially it is the equivalent of launching the IPU
  in a loop over that number of batches. This is efficient because that loop executes
  on the IPU directly.

  .. literalinclude:: device_iterations.py
    :caption: Use of device iterations
    :lines: 3-31
    :linenos:


This has a increasing effect as you set more options. The other options which can be set
are:

* Replication factor. This will replicate the model over N IPUs to allow automatic data parallelism
  across many IPUs.

  .. literalinclude:: device_iterations.py
    :caption: Use of replication factor
    :lines: 33-40
    :linenos:


* Gradient accumulation. Accumulate the gradient N times before applying it. This is needed to train with
  models expressing pipelined model parallelism using the IPU annotation. This is due to weights being
  shared across pipeline batches so gradients will be updated and used by subsequent batches out of order.

  .. literalinclude:: device_iterations.py
    :caption: Use of gradient accumulation
    :lines: 42-50
    :linenos:
