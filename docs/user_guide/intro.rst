============
Introduction
============

PopTorch is a set of extensions for PyTorch to enable PyTorch models to run directly
on Graphcore IPU hardware. PopTorch has been designed to require as few changes as
possible to your models in order to run on the IPU. However, it does have some
differences from native PyTorch execution, to get the most out of IPU hardware.

PopTorch supports executing native PyTorch models for both inference and training.
To run a PyTorch model on the IPU, you must wrap your model with either:

* :py:func:`poptorch.inferenceModel`
* :py:func:`poptorch.trainingModel`

Both of these functions accept a PyTorch model (``torch.nn.Module``) and create a
representation of the model that can be executed on the IPU hardware.

In training mode, PopTorch uses its own automatic differentiation engine
(autograd) that differs from native PyTorch.  The input model (``torch.nn.Module``)
is required to have at least one loss built into the forward pass.  PopTorch
backpropagates the gradients from the loss value(s) to update the model
parameters. This is all taken care of automatically so your training loop does not
need to call ``.backward()`` on the loss value(s) or ``.step()`` on the optimiser.

The following example shows a typical native PyTorch training loop.  The model
incorporates a loss criterion within the ``.forward()`` method, and returns the loss
value as a second output (along with the prediction).  This native PyTorch training
loop manually invokes the ``.backward()`` method to backpropagate the gradients.
The loop also manually updates the optimiser by calling the ``.step()`` method.

.. literalinclude:: poptorch_training_simple.py
  :caption: A simple example of training using PyTorch on the CPU
  :linenos:
  :start-after: simple_cpu_start
  :end-before: simple_cpu_end


Data Batching
=============

An equivalent training loop executing the model on the IPU with PopTorch is shown
below. The :py:class:`poptorch.DataLoader` is used to efficiently load data batches
on the IPU.  PopTorch follows the data batching semantics of PopART. By default,
this means you will just pass in data of the normal batch size. However, there are a
number of options provided in PopTorch which will enable more efficient data
loading. See :ref:`efficient_data_batching` for more information.

Notice that the ``torch.optim.AdamW`` optimiser is passed as an input argument to the
:py:func:`poptorch.trainingModel` wrapper which applies the optimiser algorithm
during training on the IPU.  The optimiser state is automatically managed by the
PopART framework so there is no need to call the ``.step()`` method.  Another
significant change from the native training loop is there is no ``loss.backward()``.
As mentioned above, PopTorch uses its own automatic differentiation engine and will
detect the loss value to backpropagate the gradients from.

.. literalinclude:: poptorch_training_simple.py
  :caption: Equivalent code using PopTorch to train on the IPU
  :linenos:
  :start-after: simple_ipu_start
  :end-before: simple_ipu_end


Distributed Execution
=====================

For additional scalability, you can wrap individual layers in an IPU helper to
designate the IPU to execute the layer. Using the user-provided annotations,
PopTorch will use PopART to parallelise the model over the given number of IPUs.
Additional parallelism can be expressed via a replication factor which enables you
to data-parallelise the model over additional IPUs.
See :ref:`distributed_execution` for addtional information.


Constraints
===========

PopTorch uses PyTorch's ``torch.jit.trace`` API. That means it inherits the
constraints of that API. These include:

   * Inputs must be PyTorch tensors or tuples containing PyTorch tensors.
   * ``None`` can be used as a default value for a parameter but cannot be
     explicitly passed as an input value.
   * ``torch.jit.trace`` cannot handle control flow or shape variations within the
     model. That is, the inputs passed at run-time cannot vary the control flow of the
     model or the shapes/sizes of results. If you attempt this, the graph will be
     frozen to whichever control flow path was traced as a result of the first inputs
     given to the wrapped model.

.. note:: All tensor data types and shapes must be constant for the entire dataset.

Not all PyTorch operations have been implemented by the PopTorch compiler yet.  See
:ref:`supported_ops` for a list of operators that are supported on the IPU.
Please also report any unsupported operators to support@graphcore.ai so that these
ops may be incorporated into a future release.