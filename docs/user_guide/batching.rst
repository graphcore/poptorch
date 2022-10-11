.. _efficient_data_batching:

=======================
Efficient data batching
=======================

By default, PopTorch will process the ``batch_size`` which you provided to
the :py:class:`poptorch.DataLoader`. This value is known as the micro-batch
size.

When using the other options below, the actual number of samples used per step
varies to allow the IPU(s) to process data more efficiently.

However, the effective batch size for operations which depend on it (for example the size of mini-batches, in PyTorch's terminology, when using Pytorch's `BatchNorm <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>`__ layers) will not change. All that changes is how much data is
actually sent for a single step.

.. note:: Failure to use :py:class:`poptorch.DataLoader` may result in
   accidentally changing the effective batch size for operations which depend on
   it, such as batch normalization.

You can find a detailed tutorial regarding efficient data loading, batching and tuning
relevant hyperparameters in PopTorch on Graphcore's tutorials repository: :tutorials-repo:`PopTorch tutorial: Efficient data loading  <tutorials/pytorch/efficient_data_loading>`.

poptorch.DataLoader
===================

PopTorch provides a thin wrapper around the traditional `torch.utils.data.DataLoader <https://pytorch.org/docs/1.10.0/data.html#torch.utils.data.DataLoader>`_
to abstract away some of the batch sizes calculations. If :py:class:`poptorch.DataLoader`
is used in a distributed execution environment, it will ensure that each process uses
a different subset of the dataset.

If you set the :py:class:`poptorch.DataLoader` ``batch_size`` to more than 1
then each operation in the model will process that number of elements at any
given time. Please see the usage example below.

poptorch.AsynchronousDataAccessor
=================================

To reduce host overhead you can offload the data loading process to a
separate thread by specifying :py:class:`mode=poptorch.DataLoaderMode.Async <poptorch.DataLoaderMode>` in the
:py:class:`~poptorch.DataLoader` constructor. Internally this uses an
:py:class:`~poptorch.AsynchronousDataAccessor`. Doing this allows you to reduce
the host/IPU communication overhead by using the time that the IPU is running
to load the next batch on the CPU. This means that when the IPU is finished
executing and returns to host the data will be ready for the IPU to pull in again.

.. literalinclude:: device_iterations.py
  :caption: Use of AsynchronousDataAccessor
  :start-after: data_accessor_start
  :end-before: data_accessor_end
  :emphasize-lines: 10
  :linenos:

.. warning:: Tensors being iterated over using an
  :py:class:`~poptorch.AsynchronousDataAccessor` use shared memory. You must clone
  tensors at each iteration if you wish to keep their references outside of each
  iteration.

  Consider the following example:

  .. code-block:: python
    :emphasize-lines: 5

    predictions, labels = [], []

    for data, label in dataloader:
        predictions += poptorch_model(data)
        labels += label

  The ``predictions`` list will be correct because it's producing a new tensor from the
  inputs. However, The list ``labels`` will contain identical references. This line
  would need to be replaced with the following:

  .. code-block:: python

    labels += label.detach().clone()

Rebatching iterable datasets
----------------------------

There are `two types of datasets in PyTorch <https://pytorch.org/docs/1.10.0/data.html#dataset-types>`_ : map-style datasets and iterable datasets.

As explained in the notes of PyTorch's `Data Loading Order and Sampler <https://pytorch.org/docs/1.10.0/data.html#data-loading-order-and-sampler>`_ : for
`IterableDataset <https://pytorch.org/docs/1.10.0/data.html#torch.utils.data.IterableDataset>`_ :
"When fetching from iterable-style datasets with multi-processing, the drop_last argument drops the
last non-full batch of each worker’s dataset replica."

This means that if the number of elements is naively
divided among the number of workers (which is the default behaviour) then potentially a significant number of elements will be dropped.

For example:

.. code-block:: python

  num_tensors = 100
  num_workers = 7
  batch_size = 4

  per_worker_tensors = ceil(100 / num_workers) = 15
  last_worker_tensors = 100 - (num_workers - 1) * per_worker_tensors = 10

  num_tensors_used = batch_size * (floor(per_worker_tensors / batch_size) * (num_workers - 1) + floor(last_worker_tensors / batch_size))
                   = 80

This means in this particular case 20% of the dataset will never be used. But, in general the larger the number of workers and the batch size, the more data will end up being unused.

To work around this issue PopTorch has a :py:class:`mode=poptorch.DataLoaderMode.AsyncRebatched <poptorch.DataLoaderMode>`.
PopTorch will set the ``batch_size`` in the PyTorch Dataset and DataLoader to ``1`` and will instead create the batched tensors in its worker process.

The shape of the tensors returned by the DataLoader will be the same as before, but the number of used tensors from the dataset  will increase to
``floor(num_tensors / batch_size) * batch_size`` (which means all the tensors would be used in the example above).

.. note:: This flag is not enabled by default because the behaviour is different from the upstream DataLoader.

.. _device_iterations:

poptorch.Options.deviceIterations
=================================

When training, a device iteration corresponds to one iteration of the training
loop executed on the IPU, starting with data loading, followed by the forward
and backward passes, and ending with a weight update. If
:ref:`gradient accumulation <gradient_accumulation>` is not used then if you set
:py:meth:`~poptorch.Options.deviceIterations` to `n`, PopTorch will carry out
this loop `n` times (processing `n` micro-batches) on the IPU before returning
control to the host, which will improve processing efficiency. If gradient
accumulation is used then the number of micro-batches processed will be `n`
multiplied by the value set using
:py:meth:`~poptorch.options._TrainingOptions.gradientAccumulation`.

For inference, a device iteration corresponds to data loading and the forward pass.

Note that the returned output dimensions depend on
:py:meth:`poptorch.Options.outputMode`. The default value for
:py:func:`poptorch.trainingModel` is `Final`, since you will often not need to
receive all or any of the output tensors and it is more efficient not to
receive them. Therefore, only the last batch of data will be returned to the
host under this setting. You can change this behaviour by setting the value of
:py:meth:`poptorch.Options.outputMode`. to `All`. This returns the result of
every batch to the host.

.. note:: When running an
  :py:class:`~poptorch.inferenceModel` with
  :py:class:`poptorch.PipelinedExecution`, you must set
  :py:meth:`~poptorch.Options.deviceIterations` to at least the number of
  pipeline steps.


.. literalinclude:: device_iterations.py
  :caption: Use of device iterations and batch size
  :start-after: iterations_start
  :end-before: iterations_end
  :emphasize-lines: 51, 57, 63
  :linenos:

poptorch.Options.replicationFactor
==================================

:py:meth:`~poptorch.Options.replicationFactor` will replicate the model over
multiple IPUs to allow automatic data parallelism across many IPUs.

.. literalinclude:: device_iterations.py
  :caption: Use of replication factor
  :start-after: replication_start
  :end-before: replication_end
  :emphasize-lines: 13
  :linenos:

poptorch.Options.inputReplicaGrouping
=====================================

:py:meth:`~poptorch.Options.inputReplicaGrouping` allows the input batches to
be split between groups of replicas, in a similar way to what
:py:func:`~replicaGrouping` does for weight tensors. See
:numref:`grouping_tensor_weights`.

.. _gradient_accumulation:

poptorch.Options.Training.gradientAccumulation
==============================================

You can use
:py:meth:`~poptorch.options._TrainingOptions.gradientAccumulation`
to run a number of micro-batches before updating parameters (weights) during
training. The number of gradient accumulations is equal to the number of
micro-batches (batches whose size is specified as the ``batch_size`` value
provided to the :py:class:`~poptorch.DataLoader`) which are processed between
model updates. After accumulation, PopTorch updates the model using the
gradients accumulated from processing all the batches.

.. note:: When running an :py:class:`~poptorch.inferenceModel`,  you must set
  :py:meth:`~poptorch.options._TrainingOptions.gradientAccumulation` to 1.

As mentioned in :numref:`pipelined_execution`, you need to use gradient
accumulations when training with :py:class:`poptorch.PipelinedExecution`
because the parameters can only be updated between pipeline runs.  You need to
set the number of accumulations to at least the number of pipeline stages.
However, with this value, the pipeline will switch into the "ramp-down"
period as soon as it has finished the "ramp-up" period. Using a larger number
of gradient accmumulations means that the pipeline will run at full efficiency
for longer. However, the increase in batches between parameter updates may
reduce the overall training efficiency of your model. The optimal number of
gradient accumulations is a trade off between these two factors.

.. note:: :py:meth:`~poptorch.options._TrainingOptions.gradientAccumulation`
   is only needed by :py:class:`poptorch.PipelinedExecution`. Other execution
   modes may benefit from it because the IPUs will spend less time updating
   parameters during training.


.. literalinclude:: device_iterations.py
  :caption: Use of gradient accumulation
  :start-after: gradient_acc_start
  :end-before: gradient_acc_end
  :emphasize-lines: 12
  :linenos:

In the code example below, :py:class:`poptorch.Block` introduced in
:numref:`execution_strategies` is used to divide up
a different model into disjoint subsets of layers.
These blocks can be shared among multiple parallel execution strategies.

.. literalinclude:: mnist.py
  :language: python
  :linenos:
  :start-after: annotations_start
  :end-before: annotations_end
  :emphasize-lines: 12, 14, 16, 18, 34
  :caption: A training model making use of :py:class:`poptorch.Block`

You can see the code examples of :py:class:`poptorch.SerialPhasedExecution`,
:py:class:`poptorch.PipelinedExecution`, and
:py:class:`poptorch.ShardedExecution` below.

An instance of class :py:class:`poptorch.PipelinedExecution` defines an
execution strategy that assigns layers to multiple IPUs as a pipeline. Gradient
accumulation is used to push multiple batches through the pipeline allowing
IPUs to run in parallel.

.. literalinclude:: mnist.py
  :caption: An example of different parallel execution strategies
  :language: python
  :linenos:
  :start-after: annotations_strategy_start
  :end-before: annotations_strategy_end
  :emphasize-lines: 6, 13, 19, 21


:numref:`figPipeline` shows the pipeline execution for multiple batches
on IPUs. There are 4 pipeline stages running on 4 IPUs respectively.
Gradient accumulation enables us to keep the same number of pipeline stages,
but with a wider pipeline.
This helps hide the latency, which is the total time for one item to go
through the whole system, as highlighted.

.. _figPipeline:
.. figure:: IPU-pipeline.jpg
   :width: 400

   Pipeline execution with gradient accumulation

.. _trainingOutputMode:

poptorch.Options.outputMode
==========================================

When you use a :py:func:`~poptorch.inferenceModel`, you will usually want to
receive all the output tensors. For this reason, PopTorch will return them
all to you by default. However, you can change this behaviour using
:py:func:`poptorch.Options.outputMode`.

When you use a :py:func:`~poptorch.trainingModel`, you will often not need to
receive all or any of the output tensors and it is more efficient not to
receive them. For this reason, PopTorch only returns the last batch of tensors
by default. As in the the case of ``inferenceModel``, you can change this
behaviour using :py:func:`poptorch.Options.outputMode`.

If you want to monitor training using a metric such as loss or accuracy, you
may wish to take into account all tensors. To do this with minimal or no
overhead, you can use ``poptorch.OutputMode.Sum``. For example:

 .. literalinclude:: sumAnchorReturnType.py
  :caption: A model which returns training accuracy as a tensor
  :language: python
  :linenos:
  :start-after: model_returning_accuracy_start
  :end-before: model_returning_accuracy_end

 .. literalinclude:: sumAnchorReturnType.py
  :caption: Efficient calculation of training accuracy across all batches
  :language: python
  :linenos:
  :start-after: sum_accuracy_start
  :end-before: sum_accuracy_end
