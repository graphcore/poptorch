============
Introduction
============

PopTorch is a set of extensions for PyTorch to enable PyTorch models to run directly
on the Graphcore IPU. PopTorch has been designed to require as few changes as
possible to your models in order to run on the IPU. However, it does have some
differences from native PyTorch execution, to get the most out of IPU hardware.
The `IPU Programmer's Guide <https://docs.graphcore.ai/projects/ipu-programmers-guide/>`__ provides an introduction to the IPU architecture, programming model and tools available.

PopTorch is included with the `Poplar SDK <https://docs.graphcore.ai/projects/sdk-overview/>`__.
See the `Getting Started guide <https://docs.graphcore.ai/en/latest/getting-started.html>`_ for your system for how to
install the Poplar SDK. Refer to :numref:`installation` for how to install the PopTorch wheel.

In the Graphcore software stack, PyTorch sits at the highest level of
abstraction. Poplar and PopLibs provide a software interface to operations
running on the IPU. PopTorch compiles PyTorch models into Poplar executables and
also provides IPU-specific functions.

.. figure:: pytorch-software-stack.png
    :align: center
    :width: 100%

    PyTorch, PopTorch and the Poplar software stack

PopTorch supports executing native PyTorch models for both inference and training.
To run a PyTorch model on the IPU, you must wrap your model with either:

* :py:func:`~poptorch.inferenceModel`
* :py:func:`~poptorch.trainingModel`

Both of these functions accept a PyTorch model (`torch.nn.Module <https://pytorch.org/docs/1.10.0/generated/torch.nn.Module.html#torch.nn.Module>`_) and create a
representation of the model that can be executed on the IPU hardware.

In training mode, PopTorch uses its own automatic differentiation engine
(autograd) that differs from native PyTorch.  The input model (`torch.nn.Module <https://pytorch.org/docs/1.10.0/generated/torch.nn.Module.html#torch.nn.Module>`_)
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


Data batching
=============

An equivalent training loop executing the model on the IPU with PopTorch is shown
below. The :py:class:`~poptorch.DataLoader` class is used to efficiently load data batches
on the IPU.  PopTorch follows the data batching semantics of `PopART <https://docs.graphcore.ai/projects/popart-user-guide/>`__. By default,
this means you will just pass in data of the normal batch size. However, there are a
number of options provided in PopTorch which will enable more efficient data
loading. See :numref:`efficient_data_batching` for more information.

Notice that the `torch.optim.AdamW <https://pytorch.org/docs/1.10.0/optim.html#torch.optim.AdamW>`_ optimiser is passed as an input argument to the
:py:func:`~poptorch.trainingModel` wrapper which applies the optimiser algorithm
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


Parallel and Distributed execution
==================================

To scale your models, you can enable :ref:`execution_strategies` using the
PopTorch :ref:`annotation_tools` to label or wrap individual parts of your
model and assign parts of the model to an individual IPU or execution phase.
You can also use PopTorch's :ref:`available_execution_strategies` to determine how the
model executes the phases.

Having assigned the model to run on one or more IPUs, you can add additional
parallelism with replication. Each replica represents an additional copy of the
entire model. These copies run in parallel.

PopTorch can also run across multiple hosts. This is necessary for using more
than 64 IPUs across IPU Pod systems and may be beneficial when using a smaller number
of IPUs, for example with models that involve intensive pre-processing on the CPU. We
recommend using the PopRun command-line tool and and PopDist configuration
library, which can automatically set up PopTorch to run across multiple IPU-POD
hosts. Refer to the `PopDist and PopRun User Guide
<https://docs.graphcore.ai/projects/poprun-user-guide/>`__ for more information,
including details about the installation of Horovod if you are using the MPI
communication protocol.

.. _constraints:

Constraints
===========

The following constraints apply when using PopTorch:

* All tensor data types and shapes must be constant for the entire dataset.

* As PopTorch compiles to a static graph, it cannot handle control flow
  variations within the model. This means that the inputs passed at run-time
  cannot vary the control flow of the model or the shapes or sizes of results.
  If this is attempted, the graph will be frozen to whichever control flow path
  was activated as a result of the first inputs given to the wrapped model.

* Not all PyTorch operations are implemented within the PopTorch compiler.  See
  :numref:`supported_ops` for a list of operators that are supported on the IPU.
  Please also report any unsupported operators to support@graphcore.ai so that these
  ops may be incorporated into a future release.

* Whilst any argument type can be used in the forward method, only tensor
  arguments may change between model invocations, as other types will be
  statically compiled inside the executable.


Other resources
===============

`Switching from GPUs to IPUs for Machine Learning Models <https://docs.graphcore.ai/projects/differences-ipu-gpu/>`__ provides a high-level overview of the programming changes required when switching from GPUs to IPUs and `Memory and Performance Optimisation on the IPU <https://docs.graphcore.ai/projects/memory-performance-optimisation/>`__ presents guidelines to help you develop high-performance machine learning models running on the IPU.

The Graphcore `Examples GitHub repository <https://github.com/graphcore/examples>`_ contains PopTorch applications, :tutorials-repo:`feature examples <feature_examples/pytorch>`,
:tutorials-repo:`tutorials <tutorials/pytorch>` and :tutorials-repo:`simple applications <simple_applications/pytorch>`.
Further developer resources can be found on `Graphcore's developer portal <https://www.graphcore.ai/developer>`_.
