=====================
Debugging your model
=====================

Inspecting tensors
==================

PopTorch allows you to inspect arbitrary tensors in both inference and
training models. This is very useful for debugging conditions such as
overflows, underflows or vanishing gradients.

Numerous tensors are generated during model compilation. In order to have
their values inspected, you first have to figure out their names. You can
retrieve the complete list of tensor names in your model by calling
:py:func:`~poptorch.PoplarExecutor.getTensorNames`. Note that the model
must first be compiled.

.. literalinclude:: debugging.py
  :caption: Retrieving the list of tensor names
  :start-after: tensor_names_start
  :end-before: tensor_names_end
  :emphasize-lines: 8

Anchoring tensors
=================

Once you have chosen a few tensors of interest, the next step is to create anchors.
Anchoring enables a tensor to be observed by the application without it having to
be a model output.

You can create an anchor by calling :py:func:`~poptorch.Options.anchorTensor`.
It takes two mandatory string parameters: a convenient user-defined name for
the anchor and the name of the chosen tensor. Optionally, you may specify the
anchor mode as well as the anchor return period. In order for this option setting
to take effect, they must be set before model compilation.

In the example below, two anchors are created: one for a bias gradient tensor
and one for the updated weights of a linear layer.

.. literalinclude:: debugging.py
  :caption: Anchoring tensors
  :start-after: tensor_anchor_start
  :end-before: tensor_anchor_end
  :emphasize-lines: 2, 3

Retrieving tensors
==================

The anchored tensors will be updated after every model invocation. You can
retrive their values using :py:func:`~poptorch.PoplarExecutor.getAnchoredTensor`.
The function takes a single parameter - the user-defined anchor name.

In the example below, we execute one training run and retrieve the values of
the two tensors we have anchored previously.

.. literalinclude:: debugging.py
  :caption: Anchoring tensors
  :start-after: tensor_retrieve_start
  :end-before: tensor_retrieve_end
  :emphasize-lines: 4, 5

For a more practical understanding around observing tensors, Graphcore's tutorials
repository contains a tutorial you can follow about observing tensors, using anchoring and generating a gradient histogram:
`PopTorch tutorial: Observing tensors <https://github.com/graphcore/tutorials/tree/sdk-release-2.3/tutorials/pytorch/tut4_observing_tensors>`_.
