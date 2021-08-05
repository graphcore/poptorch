========================
From PyTorch to PopTorch
========================

This page will introduce the key features that enable training on the IPU, and how they differ from native PyTorch.

Preparing your data
===================

Data loading in PyTorch is typically handled using `torch.utils.data.DataLoader <https://pytorch.org/docs/1.9.0/data.html#torch.utils.data.DataLoader>`_.

PopTorch extends PyTorch's DataLoader with a :py:class:`poptorch.DataLoader` to enable efficient data batching with respect to PopTorch's underlying machine learning framework, PopART.
Instantiation is almost identical to PyTorch, but you must remember to pass an instance of :py:class:`poptorch.Options`.

+-------------------------------------------------+
| PyTorch                                         |
+-------------------------------------------------+
| .. literalinclude:: poptorch_training_simple.py |
|   :lines: 1-5                                   |
|   :start-after: simple_cpu_start                |
|   :end-before: simple_cpu_end                   |
|   :dedent: 4                                    |
+-------------------------------------------------+
| PopTorch                                        |
+-------------------------------------------------+
| .. literalinclude:: poptorch_training_simple.py |
|   :lines: 1-9                                   |
|   :start-after: simple_ipu_start                |
|   :end-before: simple_ipu_end                   |
|   :dedent: 4                                    |
+-------------------------------------------------+

For more information about how to set :py:class:`poptorch.Options`, see :numref:`efficient_data_batching`.

Creating your model
===================

Training
--------

If you want to create a model for training on the IPU, all you need to do is instantiate a :py:func:`~poptorch.trainingModel`,
by passing your PyTorch model, :py:class:`~poptorch.Options`, and optimizer.

+-------------------------------------------------+
| PyTorch                                         |
+-------------------------------------------------+
| .. literalinclude:: poptorch_training_simple.py |
|   :lines: 7-10                                  |
|   :start-after: simple_cpu_start                |
|   :end-before: simple_cpu_end                   |
|   :dedent: 4                                    |
+-------------------------------------------------+
| PopTorch                                        |
+-------------------------------------------------+
| .. literalinclude:: poptorch_training_simple.py |
|   :lines: 11-19                                 |
|   :start-after: simple_ipu_start                |
|   :end-before: simple_ipu_end                   |
|   :dedent: 4                                    |
|   :emphasize-lines: 7-9                         |
+-------------------------------------------------+

Inference
---------

For inference, it's even easier. Just instantiate an :py:func:`~poptorch.inferenceModel` by passing your PyTorch model.

.. code-block:: python

  poptorch_model = poptorch.inferenceModel(model)

The training loop
=================

A simple training loop in PyTorch will typically consist of:

- Setting gradients to zero
- Performing a forwards pass with the model (and obtaining the loss)
- Performing the backwards pass with respect to the loss, and updating weights
- Updating the optimizer

In PopTorch, these steps are combined into a single step

+-------------------------------------------------+
| PyTorch                                         |
+-------------------------------------------------+
| .. literalinclude:: poptorch_training_simple.py |
|   :lines: 14-25                                 |
|   :start-after: simple_cpu_start                |
|   :end-before: simple_cpu_end                   |
|   :dedent: 4                                    |
+-------------------------------------------------+
| PopTorch                                        |
+-------------------------------------------------+
| .. literalinclude:: poptorch_training_simple.py |
|   :lines: 23-26                                 |
|   :start-after: simple_ipu_start                |
|   :end-before: simple_ipu_end                   |
|   :dedent: 4                                    |
+-------------------------------------------------+

Multiple/custom losses
======================

If using multiple losses, or when creating a custom loss, the final loss must be marked explicitly using :py:func:`~poptorch.identity_loss`.

+----------------------------------------------------------------------+
| PyTorch                                                              |
+----------------------------------------------------------------------+
| .. code-block:: python                                               |
|                                                                      |
|   def custom_loss(output, target)                                    |
|       loss1 = torch.nn.functional.nll_loss(x, target)                |
|       loss2 = torch.nn.functional.nll_loss(x, target) * 5.0          |
|       return loss1 + loss2                                           |
+----------------------------------------------------------------------+
| PopTorch                                                             |
+----------------------------------------------------------------------+
| .. code-block:: python                                               |
|   :emphasize-lines: 4                                                |
|                                                                      |
|   def custom_loss(output, target)                                    |
|       loss1 = torch.nn.functional.nll_loss(x, target)                |
|       loss2 = torch.nn.functional.nll_loss(x, target) * 5.0          |
|       return poptorch.identity_loss(loss1 + loss2, reduction='none') |
+----------------------------------------------------------------------+

Optimizers
==========

One important thing to note about using optimizers in PopTorch is that the optimizer state is encapsulated within the PopTorch model.
As such, any change made to the optimizer outside of the model must be followed by a call to :py:meth:`poptorch_model.setOptimizer <poptorch.PoplarExecutor.setOptimizer>`,
passing in the updated optimizer.

+-------------------------------------------------+
| PyTorch                                         |
+-------------------------------------------------+
| .. literalinclude:: poptorch_training_simple.py |
|   :lines: 27-33                                 |
|   :start-after: simple_cpu_start                |
|   :end-before: simple_cpu_end                   |
|   :dedent: 4                                    |
|   :emphasize-lines: 7                           |
+-------------------------------------------------+
| PopTorch                                        |
+-------------------------------------------------+
| .. literalinclude:: poptorch_training_simple.py |
|   :lines: 28-36                                 |
|   :start-after: simple_ipu_start                |
|   :end-before: simple_ipu_end                   |
|   :dedent: 4                                    |
|   :emphasize-lines: 8-9                         |
+-------------------------------------------------+

.. note:: PopTorch also provides its own set of optimizers that can be accessed via ``poptorch.optim``.
   These are wrapper classes which have several advantages over the native PyTorch optimizers. They embed constant attributes
   for performance/memory savings and allow you to specify additional parameters such as loss scaling and velocity scaling.
   See :numref:`optimizers` for more information.

Going further
=============

For a more detailed tutorial on getting started with PopTorch, you can find the `Pytorch basics tutorial <https://github.com/graphcore/tutorials/tree/sdk-release-2.2/tutorials/pytorch/tut1_basics>`_ which walks through training an MNIST model on the IPU.
