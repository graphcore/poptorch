Features
========

Model wrapping functions
------------------------

The basis of PopTorch integration comes from these two model wrapping functions.

.. py:function:: trainingModel(model, device_iterations, gradient_accumulation=1, replication_factor=1, profile=False, trace_model=True, loss=None, optimizer=None)

    Create a PopTorch training model, from a PyTorch model, to run on IPU
    hardware in training mode. See :ref:`efficient_data_batching` for a full
    explanation of how ``device_iterations`` greater than 1, ``gradient_accumulation``, and
    ``replication_factor`` interact with the output and input sizes.

    :param nn.Module model: A PyTorch model.
    :param int device_iterations: The number of iterations the device should run over
                                  the data before returning to the user.
    :param int gradient_accumulation: The level of gradient accumulation to perform.
                                      This must be equal to or greater than the number of pipelined IPU stages.
    :param int replication_factor: The number of replicas to replicate this model over. This will stack
                                   with pipelines so a model pipelined over two IPUs with a replication
                                   factor of 4 will run over 8 IPUs.
    :param nn.Module loss: A PyTorch loss function to be added to the end of the model and fed into the autograd.
                           See :any:`identity_loss()` for an example of custom losses.
    :param optim.Optimizer: The optimizer to apply during training. Only ``optim.SGD`` is supported.
    :param bool profile: Enable generation of graph reports by Poplar.
    :param bool trace_model: If False, PopTorch will compile the model using ``torch.jit.script`` rather than
                             ``torch.jit.trace`` (script is experimental, default True).
    :return: The :any:`PoplarExecutor` wrapper to use in place of ``model``.


.. literalinclude:: trainingModel.py
    :language: python
    :caption: An example of the use of ``trainingModel()``
    :emphasize-lines: 14-18

.. py:function:: inferenceModel(model, device_iterations=1,replication_factor=1, profile=False, trace_model=True)

    Create a PopTorch inference model, from a PyTorch model, to run on IPU hardware in inference mode.
    See :ref:`efficient_data_batching`  for a full
    explanation of how ``device_iterations`` greater than 1, ``gradient_accumulation``, and
    ``replication_factor`` interact with the output and input sizes.

    :param nn.Module model: A PyTorch model.
    :param int device_iterations: The number of iterations the device should run over the data before returning to the user.
    :param int replication_factor: The number of replicas to replicate this model over.
                                   This will stack with pipelines so a model pipelined over two IPUs with a replication
                                   factor of 4 will run over 8 IPUs.
    :param bool profile: Enable generation of graph reports by Poplar.
    :param bool trace_model: If False, PopTorch will compile the model using ``torch.jit.script`` rather than ``torch.jit.trace``
                            (script is experimental, default True).
    :return: The :any:`PoplarExecutor` wrapper to use in place of ``mode``.

.. literalinclude:: inference.py
    :caption: An example of the use of ``inferenceModel()``
    :linenos:
    :emphasize-lines: 15

The above functions return a ``PoplarExecutor`` class which is just a wrapper around the provided model.

.. py:class:: PoplarExecutor

   This class should not be created directly but is a wrapper around the model
   that was passed into :any:`inferenceModel` or :any:`trainingModel`. It only has a few methods
   which can be used to interface with the IPU.

   .. py:method:: __call__(*args, **kwargs)

      Calls the wrapped model with the given tensors. Inputs must be tensors or
      tuples/lists of tensors. Will compile for IPU on the first invocation.

   .. py:method:: copyWeightsToHost()

      Updates the parameters used in model with the weights stored on device. (The weights in ``model.parameters()``)

   .. py:method:: copyWeightsToDevice()

      Copies the weights from ``model.parameters()`` to the IPU device. Implicitly called on first call.

   .. py:method:: setOptimizer(optimizer)

      Sets the optimiser for a training model. Will overwrite the previous one. Only ``optim.SGD`` is supported.



Pipeline annotator
------------------

You can use the ``IPU`` wrapper class to define model parallelism. This wraps
layers in a model so you can designate them to be placed on specific IPUs in a
pipeline.

.. py:class:: IPU(ipu_id, layer_to_call=None)

    Runs a layer on a specified IPU. All layers after this layer will also run on the same IPU until another IPU wrapper is encountered. The IPUs will be "pipelined" where each IPU is executing one stage of the operation, as the previous
    IPU is executing a previous stage on the next batch and subsequent IPUs are executing subsequent stages on previous batches.

    :param int ipu_id: The id of the IPU to run on. All subsequent layers of the
                        network will run on this IPU until another layer is wrapped. By default all
                        layers will be on IPU 0 until the first pipeline annotation is encountered.

    :param layer_to_call: The layer to run on the specified IPU.


.. literalinclude:: pipeline_simple.py
   :lines: 3-33
   :emphasize-lines: 15-16, 19-20, 23-24
   :caption: Annotations can be attached to layers in existing models.

.. literalinclude:: pipeline_simple.py
    :lines: 37-52
    :emphasize-lines: 6, 10, 14
    :caption: PopTorch also supports annotating the model directly. Both forms can be used interchangeably.


Custom ops
----------

Helper operations to be used within a model.

.. py:class:: ipu_print_tensor(tensor_to_print)

    Adds a tensor to be printed on the IPU. When this point is reached in execution the tensor will be copied back to host and printed.
    The operation is an identity operation and it will return the exact same tensor. That tensor should then be used to make sure the print
    isn't eliminated. When this operation is called in the backward pass it will print the gradient of the tensor.

    :param ipu_print_tensor: The tensor to print.

    .. literalinclude:: api.py
        :lines: 3-9


.. py:function:: poptorch.identity_loss(loss, reduction="none")

  Marks this operation as being part of the loss calculation and, as such, will back-propagate through it in the PopTorch autograd. This enables multiple losses and custom losses.

  :param tensor loss: The calculated loss.

  :param string reduction: Reduce the loss output as per PyTorch loss semantics. Supported values are:

    * "none": Don't reduce

    * "sum": Sum the losses.

    * "mean": Take the mean of the losses.


  .. literalinclude:: api.py
    :lines: 11-21
