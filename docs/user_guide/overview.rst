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
    :lines: 3-
    :emphasize-lines: 21

poptorch.inferenceModel
-----------------------

.. autofunction:: poptorch.inferenceModel

.. literalinclude:: inference.py
    :language: python
    :caption: An example of the use of :py:func:`poptorch.inferenceModel`
    :linenos:
    :lines: 3-
    :emphasize-lines: 14


poptorch.PoplarExecutor
-----------------------

.. autoclass:: poptorch.PoplarExecutor
   :special-members: __call__
   :members:


Pipeline annotator
==================

You can use the ``IPU`` wrapper class to define model parallelism in a PopTorch multi-IPU
device. Conceptually this is collecting the layers of a model into pipeline stages
to be run on specific IPUs. However, as there is a 1:1 mapping of pipeline stages
to IPUs, we simply use an IPU index to declare where the layer will be run.

.. autoclass:: poptorch.IPU
   :special-members: __init__

.. literalinclude:: pipeline_simple.py
    :language: python
    :linenos:
    :lines: 3-34
    :emphasize-lines: 15, 18, 21
    :caption: Annotations can be attached to layers in existing models.


.. literalinclude:: pipeline_simple.py
    :language: python
    :linenos:
    :lines: 77-
    :emphasize-lines: 17, 20, 24
    :caption: PopTorch also supports annotating the model directly. Both forms can be used interchangeably.


Custom ops
==========

Helper operations to be used within a model.

poptorch.ipu_print_tensor
-------------------------

.. py:class:: ipu_print_tensor(tensor_to_print)

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
    :lines: 7-18
    :emphasize-lines: 10


poptorch.identity_loss
----------------------

.. autofunction:: poptorch.identity_loss

.. literalinclude:: api.py
  :language: python
  :linenos:
  :lines: 25-43
  :emphasize-lines: 5
