============
Introduction
============

PopTorch is a set of extensions for PyTorch to enable PyTorch models to run directly
on Graphcore IPU hardware.

PopTorch supports both inference and training. To run a model on the IPU you wrap your
existing PyTorch model in either a PopTorch inference wrapper or a PopTorch training
wrapper. You can provide further annotations to partition the model across multiple IPUs.

You can wrap individual layers in an IPU helper to designate which IPU they
should go on. Using the user-provided annotations, PopTorch will use PopART to parallelise
the model over the given number of IPUs. Additional parallelism can be expressed via
a replication factor which enables you to data-parallelise the model over more
IPUs.

PopTorch has been designed to require fewer manual alterations to your models in order to run them
on IPU. However, it does have some differences from native PyTorch execution.

* In training mode, PopTorch has its own autograd and will take the model, optimizer, and
  user-provided loss function and will perform the full forward and backward pass under the
  hood.

.. literalinclude:: poptorch_training_simple.py
  :caption: A simple example of training using PyTorch on the CPU
  :linenos:
  :lines: 96-128

.. literalinclude:: poptorch_training_simple.py
  :caption: Equivalent code using PopTorch to train on the IPU
  :linenos:
  :lines: 54-89

* Under the hood PopTorch uses the ``torch.jit.trace`` API. That means it
  inherits the constraints of that API. These include:

   * Inputs must be Torch tensors or tuples/lists containing Torch tensors.
   * ``None`` can be used as a default value for a parameter but cannot be expliticly passed as an input value.
   * ``torch.jit.trace`` cannot handle control flow or shape variations within the model. That is, the inputs passed at run-time cannot vary the control flow of the model or the shapes/sizes of results.
   * Not all PyTorch operations have been implemented by the backend yet.

* PopTorch follows PopART's data batching semantics. By default this means you will just pass in data
  of the normal batch size. However, there are a number of options provided in PopTorch which will enable
  more efficient data loading. See :ref:`efficient_data_batching` for more information.
