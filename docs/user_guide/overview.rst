========
Features
========

.. contents::
  :local:

Options
=======

The compilation and execution on the IPU can be controlled using :class:`poptorch.Options`:

See :ref:`efficient_data_batching`  for a full
explanation of how ``device_iterations`` greater than 1, ``gradient_accumulation``, and
``replication_factor`` interact with the output and input sizes.

.. autoclass:: poptorch.Options
   :members:

.. autoclass:: poptorch.options._DistributedOptions
   :members:

.. autoclass:: poptorch.options._JitOptions
   :members:

.. autoclass:: poptorch.options._TrainingOptions
   :members:

.. autoclass:: poptorch.options._PopartOptions
   :members:

.. autoclass:: poptorch.options._TensorLocationOptions
   :members:

Model wrapping functions
========================


The basis of PopTorch integration comes from these two model wrapping functions.

poptorch.trainingModel
----------------------

.. autofunction:: poptorch.trainingModel

.. literalinclude:: trainingModel.py
    :language: python
    :caption: An example of the use of :py:func:`poptorch.trainingModel`
    :linenos:
    :emphasize-lines: 21
    :start-after: training_model_start

poptorch.inferenceModel
-----------------------

.. autofunction:: poptorch.inferenceModel

.. literalinclude:: inferenceModel.py
    :language: python
    :caption: An example of the use of :py:func:`poptorch.inferenceModel`
    :linenos:
    :start-after: inference_model_start
    :emphasize-lines: 14


poptorch.PoplarExecutor
-----------------------

.. autoclass:: poptorch.PoplarExecutor
   :special-members: __call__
   :members:

.. note:: The ``PoplarExecutor`` will implicitly keep in sync the parameters
  of the source Torch model and the PopTorch model(s)
  however weights need to be explicitly copied if the
  model is trained on the CPU and inference is run on the IPU.

  .. code-block:: python

    model = Model()
    poptorch_train = poptorch.trainingModel(model)
    poptorch_inf = poptorch.inferenceModel(model)

    train(poptorch_train)
    torch.save(model.state_dict(), "model.save") # OK
    validate(poptorch_inf) # OK
    validate(model) # OK

    train(model)
    # Explicit copy needed
    poptorch_inf.copyWeightsToDevice()
    validate(poptorch_inf)


Pipeline annotator
==================

You can use the :py:class:`poptorch.BeginBlock` or :py:class:`poptorch.Block` wrapper class to define model parallelism
in a PopTorch multi-IPU device. Conceptually this is collecting the layers of a model
into a phase which can be mapped onto a specific IPU. By default, these will execute
using a pipelined model parallel execution strategy.

We expose an additional argument `phase_id` through this API. This will be passed down to
PopART and when used in conjunction with PopART session options can override the default
pipeline execution strategy. Refer to the PopART user guide to enable other execution modes.

.. autoclass:: poptorch.Block
   :special-members: __init__, useAutoId

.. autoclass:: poptorch.BeginBlock
   :special-members: __init__

.. autoclass:: poptorch.PipelinedExecution
   :special-members: __init__
   :inherited-members:

.. autoclass:: poptorch.AutoStage

.. autoclass:: poptorch.Stage
   :special-members: __init__

.. autoclass:: poptorch.ShardedExecution
   :inherited-members:

.. autoclass:: poptorch.Phase
   :special-members: __init__

.. autoclass:: poptorch.ParallelPhasedExecution
   :special-members: __init__
   :inherited-members:

.. autoclass:: poptorch.SerialPhasedExecution
   :special-members: __init__
   :inherited-members:

.. autoclass:: poptorch.Liveness

.. literalinclude:: pipeline_simple.py
    :language: python
    :linenos:
    :start-after: annotations_start
    :end-before: annotations_end
    :emphasize-lines: 15-16, 19-20, 23-24
    :caption: Annotations can be attached to layers in existing models.


.. literalinclude:: pipeline_simple.py
    :language: python
    :linenos:
    :start-after: annotations_inline_start
    :end-before: annotations_inline_end
    :emphasize-lines: 19, 22, 26
    :caption: PopTorch also supports annotating the model directly. Both forms can be used interchangeably.


Custom ops
==========

Helper operations to be used within a model.

poptorch.ipu_print_tensor
-------------------------

.. py:class:: ipu_print_tensor(tensor_to_print, optional_title)

    Adds a tensor to be printed on the IPU. When this is executed the tensor
    will be copied back to host and printed.

    When this operation is called in the backward pass it
    will print the gradient of the tensor.

    The operation is an identity operation and it will return the exact same
    tensor. The returned tensor should be used in place of the original tensor,
    in the rest of the program to make sure that the print operation isn't optimised away.

    For example if the original code looks like this:

    .. code-block:: python

      def forward(self, c, d, b)
        a = c + d
        return a + b

    And you want to print the value of `a`.
    If you do:

    .. code-block:: python

      def forward(self, c, d, b)
        a = c + d
        poptorch.ipu_print_tensor(a)
        return a + b

    Optionally, you may add a second string parameter to be used as a title.

    .. code-block:: python

    def forward(self, c, d, b)
        a = c + d
        poptorch.ipu_print_tensor(a, "summation"))
        return a + b

    The result of `ipu_print_tensor` is not used, therefore it will be optimised out by the
    graph optimiser and `a` will not be printed.

    Instead you should do:

    .. code-block:: python

      def forward(self, c, d, b)
        a = c + d
        x = poptorch.ipu_print_tensor(a)
        return x + b

    .. warning::
       In order for the print operation to not be optimised out by the graph
       optimiser, you must use the output of the print.

    :param ipu_print_tensor: The tensor to print.
    :returns: The input unchanged.

.. literalinclude:: api.py
    :language: python
    :linenos:
    :start-after: print_tensor_start
    :end-before: print_tensor_end
    :emphasize-lines: 10


poptorch.identity_loss
----------------------

This function is used to implement custom losses. This takes in a single torch tensor
and will backpropagate a gradient of ones through it.

.. warning::
   Passing a pytorch loss function or another identity_loss to this function is not currently
   supported. Multiple losses must be implemented via composite torch ops.

.. autofunction:: poptorch.identity_loss

.. literalinclude:: api.py
  :language: python
  :linenos:
  :start-after: identity_start
  :end-before: identity_end
  :emphasize-lines: 5

Half / float 16 support
=======================

PopTorch supports the half-precision floating point (float 16) format.
You can convert models to float 16 by using a Module's .half() method and by using float 16 rather than float 32 tensors for the input.
(You can convert a tensor to float 16 using ``tensor = tensor.half()``)

.. literalinclude:: inferenceModel.py
    :language: python
    :caption: How to run a model using half precision
    :linenos:
    :start-after: inference_half_start
    :end-before: inference_half_end
    :emphasize-lines: 1, 2

Because PopTorch relies on the ``torch.jit.trace`` API, it is limited to tracing operations which run on the CPU.
Many of these operations do not support float 16 inputs.
To allow the full range of operations, PopTorch converts all float 16 inputs to float 32 before tracing and then restores the inputs to float 16 as part of the canonicalization process.
Some operations may result in the model running in float 32 where float 16 would be expected, or vice versa (see :ref:`float_16_op_support` for full details).

Profiling
=========

You can profile a graph produced by PopTorch for analysis using the PopVision Graph Analyser, which can be downloaded from the Graphcore support portal.
To do this, use the :ref:`POPLAR_ENGINE_OPTIONS<profiling_env>` environment variable.

Environment variables
=====================

Logging level
-------------
PopTorch uses the following levels of logging:
  * ``OFF``: No logging.
  * ``ERR``: Errors only.
  * ``WARN``: Warnings and errors only.
  * ``INFO``: Info, warnings and errors. (Default)
  * ``DEBUG``: Adds some extra debugging information.
  * ``TRACE`` and ``TRACE_ALL``: Trace everything inside PopTorch.

The ``POPTORCH_LOG_LEVEL`` environment variable can be used to set the logging level:

.. code-block:: bash

  export POPTORCH_LOG_LEVEL=DEBUG

.. _profiling_env :

Profiling
---------

When running programs using PopTorch, you can enable profiling by using the ``POPLAR_ENGINE_OPTIONS`` environment variable used by Poplar.
In order to capture the reports needed for the PopVision Graph Analyser you only need to set ``POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true"}'``:

.. code-block:: bash

  export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true"}'

By default, report files are output to the current working directory. You can specify a different output directory by setting ``autoReport.directory``, for example:

.. code-block:: bash

  export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./tommyFlowers"}'

For more options, please refer to the `PopVision Graph Analyser User Guide <https://docs.graphcore.ai/projects/graphcore-popvision-user-guide>`_.

IPU Model
---------

By default PopTorch will try to attach to a physical IPU.
If instead you want to use the model, you can do so by setting ``POPTORCH_IPU_MODEL`` to ``1``:

.. code-block:: bash

  export POPTORCH_IPU_MODEL=1

Please see the `Poplar and PopLibs User Guide <https://docs.graphcore.ai/projects/poplar-user-guide>`_ for the limitations of the IPU Model.

Wait for an IPU to become available
-----------------------------------

By default if you try to attach to an IPU but all the IPUs in the system are
already in use, an exception will be raised.
If you would rather wait for an IPU to become available, you can do so by setting ``POPTORCH_WAIT_FOR_IPU`` to ``1``.

.. code-block:: bash

  export POPTORCH_WAIT_FOR_IPU=1
