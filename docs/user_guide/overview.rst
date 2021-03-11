========
Features
========

.. contents::
  :local:

Options
=======

The compilation and execution on the IPU can be controlled using :class:`poptorch.Options`.
Broadly speaking, the functionality provided can be broken down in the following:

#. Options related to half precision (see :class:`poptorch.options._PrecisionOptions`)
#. Management of the training process (see :class:`poptorch.options._TrainingOptions`)
#. Control of distributed execution environments
   (see :class:`poptorch.options._DistributedOptions`)
#. Location of tensors (see: :class:`poptorch.options._TensorLocationOptions` and
   :class:`poptorch.TensorLocationSettings`)
#. Options relevant to the Torch JIT compiler
   (see :class:`poptorch.options._JitOptions`)

See :ref:`efficient_data_batching`  for a full
explanation of how ``device_iterations`` greater than 1, ``gradient_accumulation``, and
``replication_factor`` interact with the output and input sizes.

You can choose to use the IPU model or the real IPU hardware
via :py:class:`poptorch.Options.useIpuModel`.


Model wrapping functions
========================


The basis of PopTorch integration comes from these two model wrapping functions.

poptorch.trainingModel
----------------------

This function wraps around a PyTorch model, yielding a PopTorch model that may
be run on the IPU in training mode. See :py:func:`poptorch.trainingModel` for a
complete reference.

.. literalinclude:: trainingModel.py
    :language: python
    :caption: An example of the use of :py:func:`poptorch.trainingModel`
    :linenos:
    :emphasize-lines: 22
    :start-after: training_model_start

poptorch.inferenceModel
-----------------------

This function wraps around a PyTorch model, yielding a PopTorch model that can
be run on the IPU in inference mode. See :py:func:`poptorch.trainingModel` for
a complete reference.

.. literalinclude:: inferenceModel.py
    :language: python
    :caption: An example of the use of :py:func:`poptorch.inferenceModel`
    :linenos:
    :start-after: inference_model_start
    :emphasize-lines: 14


poptorch.PoplarExecutor
-----------------------

This class should not be created directly but is a wrapper around the model
that was passed into :py:func:`~poptorch.inferenceModel` or :py:func:`~poptorch.trainingModel`.
It only has a few methods which can be used to interface with the IPU.

The :py:class:`~poptorch.PoplarExecutor` will implicitly keep in sync the parameters
of the  source PyTorch model and the PopTorch model(s). However, weights need to
be  explicitly copied if the model is trained on the CPU and inference is run on
the IPU.

See :py:class:`~poptorch.PoplarExecutor` for a complete description of the IPU interface
functionality.

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

.. _parallel_execution:

poptorch.isRunningOnIpu
-----------------------

One useful utility function is :py:func:`poptorch.isRunningOnIpu`. This
returns ``True`` when executing on the IPU and ``False`` when executing
the model outside IPU scope. This allows for different code paths within
the model.

A common use case is executing equivalent code to a PopART custom operator
when running on CPU. For example:

  .. code-block:: python

    class Network(torch.nn.Module):
      def forward(self, x, y):
          if poptorch.isRunningOnIpu():
              # IPU path
              return my_custom_operator(x, y)
          else:
              # CPU path
              return my_torch_implementation(x,y)

Parallel execution
==================

This section demonstrates multi-IPU strategies for parallel execution in
PopTorch.
We recommended that you start such parallel programming from
PopTorch code that is working properly on a single IPU.

There are four kinds of execution strategies in total to run a model on a
multi-IPU device:
:py:class:`poptorch.ShardedExecution`,
:py:class:`poptorch.PipelinedExecution`,
:py:class:`poptorch.SerialPhasedExecution`.
and :py:class:`poptorch.ParallelPhasedExecution`.
These execution strategies are set through
:py:func:`poptorch.Options.setExecutionStrategy`.
The default execution strategy is :py:class:`poptorch.PipelinedExecution`.
In the following,
we first introduce the general APIs that will be applied to all four
parallel execution strategies.
Finally, we explain the four strategies with examples.

By default, PopTorch will not let you run the model if the number of IPUs is
not a power of 2.
For this reason, it is preferable to annotate the model so that the number of
IPUs used is a power of 2.
However, you can also enable :func:`poptorch.Options.autoRoundNumIPUs` to
automatically round up the number of IPUs reserved to a power of 2, with the
excess being reserved but idle.
This option is not enabled by default to prevent unintentional overbooking of
IPUs.



Annotation tools
----------------

poptorch.Block and poptorch.BeginBlock
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:class:`poptorch.BeginBlock` and :py:class:`poptorch.Block` are wrapper
classes used to define model parallelism in a multi-IPU device. They partition
models into "blocks" that will be executed on different IPUs.

You can use :py:class:`poptorch.Block` to define a scope in the context of the
model.

In the example below, all layers before ``model.bert.encoder.layer[0]`` will be
put on IPU 0 and all layers from ``model.bert.encoder.layer[0]`` onwards
(inclusive) will be on IPU 1.

.. literalinclude:: pipeline_simple.py
    :language: python
    :linenos:
    :start-after: annotations_start
    :end-before: annotations_end
    :emphasize-lines: 37-38, 41-42, 45-46
    :caption: Annotating existing layers.

:py:class:`poptorch.BeginBlock` is an annotation defined outside the
model, and applied to current and onward layers. Both forms can be used
interchangeably.

.. literalinclude:: pipeline_simple.py
    :language: python
    :linenos:
    :start-after: annotations_inline_start
    :end-before: annotations_inline_end
    :emphasize-lines: 16, 19, 22, 26
    :caption: Annotating a model directly.

Either annotation is enough to enable parallel execution in the simple cases.
By default, the layers before the first :py:class:`poptorch.BeginBlock` will be
placed on IPU 0.

Both :py:class:`poptorch.BeginBlock` and :py:class:`poptorch.Block`
need to follow a set of rules:

* All the layers must be declared inside a :py:class:`poptorch.Block` scope.
  It is to avoid missing annotation. :py:class:`poptorch.BeginBlock`
  doesn't have the same constraint because all the layers called after will
  automatically be added to the last :py:class:`poptorch.BeginBlock`.
* Please note that PopTorch needs to reserve IPUs in powers of 2 or
  multiples of 64. You are advised to configure your model accordingly
  to take full advantage of the IPUs available. However, if you need to run
  with a different number of IPUs, you can use
  ``poptorch.Options().autoRoundNumIPUs(True)`` to allow
  PopTorch to reserve more IPUs than the model specifies.
* Unused or dead layers should NOT be included in any
  :py:class:`poptorch.BeginBlock` or :py:class:`poptorch.Block`.
* If layer A happens before layer B inside the model and each layer has
  a :py:class:`poptorch.BeginBlock` associated with it,
  you need to write :py:class:`poptorch.BeginBlock` for layer A before
  :py:class:`poptorch.BeginBlock` for layer B.

Failing to obey above rules will result in compilation errors.


poptorch.Stage and poptorch.AutoStage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conceptually :py:class:`poptorch.BeginBlock` or
:py:class:`poptorch.Block` collects the
layers of a model into a :py:class:`poptorch.Stage`,
multiple stages can be combined into a :py:class:`poptorch.Phase`,
and multiple phases form a parallel execution strategy.

poptorch.Stage
""""""""""""""

:py:class:`poptorch.Stage` defines some layers of model to run on one IPU.
It can be made of one or more blocks created using
:py:class:`poptorch.BeginBlock` or :py:class:`poptorch.Block`
and identified by their ``user_id``.
Consecutive layers in a model can be defined either in the same
:py:class:`poptorch.Stage` or consecutive stages.
Whether stages run in parallel or sequentially depends on specific
parallel execution strategies.

Internally, each operation in a model is assigned a ``stage_id``
through :py:class:`poptorch.Stage`.

poptorch.AutoStage
""""""""""""""""""

You can use :py:class:`poptorch.AutoStage` if you don't want to
specify :py:class:`poptorch.Stage` by hand.
It will assign one :py:class:`poptorch.Stage`
per :py:class:`poptorch.BeginBlock` or :py:class:`poptorch.Block`.

By default ``poptorch.AutoStage.SameAsIpu`` is in use, which means the
`stage_id` of :py:class:`poptorch.Stage` will be set to the ``ipu_id``
specified for the :py:class:`poptorch.BeginBlock` or
:py:class:`poptorch.Block`.
Please note that `stage_id` must be ascending in
:py:class:`poptorch.PipelinedExecution`.
Let's use the code example above.
If your blocks "0", "1", and "2" are assigned to IPU 0, 1, and 0.
Then the :py:class:`poptorch.Block`
"2" will be assigned ``stage_id`` 0. This will make
the compiler fail to
schedule the last two stages "1" and "2" due to a conflict:

* The model implies "1" should run earlier than "2".
* their `stage_id` values suggest "2" should run earlier than "1".

When ``poptorch.AutoStage.AutoIncrement`` is in use, each new
:py:class:`poptorch.BeginBlock` or
:py:class:`poptorch.Block` will be assigned an automatically incremented
``stage_id``.
In the previous example the last stage would be assigned ``stage_id`` 2 and
the compilation would succeed.

poptorch.Phase
^^^^^^^^^^^^^^

:py:class:`poptorch.Phase` defines a processing unit of phased execution.
It may contain one or more :py:class:`poptorch.Stage`.
:py:class:`poptorch.Phase` is only used in
:py:class:`poptorch.SerialPhasedExecution` and
:py:class:`poptorch.ParallelPhasedExecution`.
It is not used in
:py:class:`poptorch.ShardedExecution` and
:py:class:`poptorch.PipelinedExecution`.

  .. code-block:: python

    with poptorch.Block("A"):
        layer()
    with poptorch.Block("B"):
        layer()
    p = Phase(poptorch.Stage("A").ipu(0), poptorch.Stage("B").ipu(1))

In the code snippet above, "A" and "B" will run in parallel on IPU 0 and 1
simultaneously since they are placed in two stages. They will run
sequentially on one IPU if they are placed in a single stage.


Advanced annotation with strings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use Python strings to represent the `user_id` and `ipu_id` for a
:py:class:`poptorch.Block` or
:py:class:`poptorch.BeginBlock`.
Since strings are evaluated at runtime,
they allow for a dynamic number of stages and phases.
Here is an example below to use formatted strings(f-strings) in
:py:class:`poptorch.ParallelPhasedExecution`.

Inside the code example below, there are two lines that f-strings are
used in the ``forward()`` class.
One is ``f"phase{phase}_ipu{ipu}"`` at Line 25,
where ``phase`` is
0, 1, 1, 2, 3, 3, 4, 5, and 5 respectively,
and ``ipu`` ranges from 0 to 1.
The total number of instances for this f-string is 12 due to
6 phases and 2 IPUs.
The other is ``f"phase{N*2-1}_ipu1"`` at Line 32,
where ``phase`` is 5 and ``ipu`` is 1.
When defining :py:class:`poptorch.Stage`,
four f-strings are used where ``n`` ranges from 0 to 2
at Line 46-47 and 50-51:

* ``f"phase_{2*n}_ipu0"``
* ``f"phase{2*n}_ipu1"``
* ``f"phase_{2*n+1}_ipu0"``
* ``f"phase{2*n+1}_ipu1"``

They refer to ``phase`` 0, 2, 4 and 1, 3, 5, with ``ipu0`` and ``ipu1``
respectively.
So all these 12 f-strings are defined in :py:class:`poptorch.BeginBlock`,
and used in :py:class:`poptorch.Stage` dynamically. They match exactly.

.. literalinclude:: phased_execution.py
  :caption: An example of parallel phased execution
  :language: python
  :linenos:
  :start-after: annotations_start
  :end-before: annotations_end
  :emphasize-lines: 25, 32, 47-48, 51-52


Parallel execution strategies
-----------------------------

With the above APIs as building blocks, we can set execution strategies
using the four kinds of execution modes, as shown below.
Note that the same annotation can be used for each of them.
They only differ in the method of parallelisation and tensor locations.

.. _sharded_execution:

poptorch.ShardedExecution
^^^^^^^^^^^^^^^^^^^^^^^^^

In this strategy, each IPU
will sequentially execute a distinct part of the model.
A single unit of processing :py:class:`poptorch.ShardedExecution` is a
shard.
A shard is specified using :py:class:`poptorch.Stage`,
or if no :py:class:`poptorch.Stage` is specified,
the `user_id` passed by
:py:class:`poptorch.BeginBlock` or :py:class:`poptorch.Block` is used.
Each shard is executed sequentially on a single IPU.
Multiple shards can be placed on multiple IPUs.
However, only one IPU is used at a time, while
the other IPUs are idle.
If an IPU is allocated to run consecutive stages,
PopART will merge consecutive stages into one on the same IPU.
Weights and activations will use the on-chip memory of the IPUs.
Layers sharing weights need to be placed on the same IPU.

:py:class:`poptorch.ShardedExecution` can be useful
for processing a single sample or debugging.
Overall it has low efficiency since only one IPU is used at a time.


poptorch.PipelinedExecution
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the default execution strategy.
It extends :ref:`sharded_execution` with parallel execution on multiple
IPUs.

Parallelisation in :py:class:`poptorch.PipelinedExecution`
requires :py:meth:`~poptorch.Options.deviceIterations`
and :py:meth:`~poptorch.options._TrainingOptions.gradientAccumulation`.
as explained in :ref:`efficient_data_batching`.
After one :py:class:`poptorch.Stage` is finished with processing a batch
on one IPU, it starts immediately processing the next batch.
This creates a pipeline where multiple batches are processed in parallel.
An IPU can only start its own :py:class:`poptorch.Stage` of a batch if
its previous :py:class:`poptorch.Stage` of the current batch is processed.
Hence, all IPUs will be occupied after a warm-up period.
A cool-down period is required to aggregate the results and apply weight
changes.


Phased execution
^^^^^^^^^^^^^^^^

:py:class:`poptorch.ParallelPhasedExecution` and
:py:class:`poptorch.SerialPhasedExecution` have the following
features in common:

* A portion of the weights and activations are transferred to and from
  streaming memory, before and after each phase.
* If the desired weights and activations are already stored in an IPU
  of the same group of IPUs,
  intra-phase cross-IPU copies can replace the copies
  to and from streaming memory.
* This specific portion is needed by the layers of the model wrapped in
  :py:class:`poptorch.BeginBlock` or :py:class:`poptorch.Block` in current
  :py:class:`poptorch.Phase`.
* They both trade off some performance
  for larger models with higher memory needs.
* Any number of phases is allowed.
* The number of stages in each :py:class:`poptorch.Phase`
  should match the number of IPUs in each group of IPUs.
* Stages inside each :py:class:`poptorch.Phase` can run in parallel.

Although you only define the :py:class:`poptorch.Phase` for forward passes,
the corresponding phases for backward passes are created correspondingly.
The order of phased execution for backward passes won't change
but you can decide whether a phase is shared by both
forward and backward passes. In other words, you decide whether to avoid
a memory transfer of a portion of the weights and activations.

poptorch.SerialPhasedExecution
""""""""""""""""""""""""""""""

In :py:class:`poptorch.SerialPhasedExecution`,
phases execute on a single group of IPUs sequentially.

  .. code-block:: python

    strategy = poptorch.SerialPhasedExecution([
      poptorch.Phase(poptorch.Stage("A"), poptorch.Stage("A2")),
      poptorch.Phase(poptorch.Stage("B"), poptorch.Stage("B2")),
      poptorch.Phase(poptorch.Stage("C"), poptorch.Stage("C2"))])

    strategy.phase(0).ipus(0,1)
    strategy.phase(1).ipus(0,1)
    strategy.phase(2).ipus(0,1)

    opts.setExecutionStrategy(strategy)

The code above causes all phases to run serially on IPUs 0 and 1.

poptorch.ParallelPhasedExecution
""""""""""""""""""""""""""""""""

In :py:class:`poptorch.ParallelPhasedExecution`,
phases are executed in parallel alternating between two groups of IPUs.
Even phases must run on even IPUs and odd phases on odd IPUs.
Inter-phase cross-IPU copies can replace the memory transfers to and from
the streaming memory, if the desired weights and activations are already
available in another group of IPUs.

  .. code-block:: python

    strategy = poptorch.SerialPhasedExecution([
      poptorch.Phase(poptorch.Stage("0"), poptorch.Stage("1")),
      poptorch.Phase(poptorch.Stage("2"), poptorch.Stage("3")),
      poptorch.Phase(poptorch.Stage("4"), poptorch.Stage("5"))])

    strategy.phase(0).ipus(0,2)
    strategy.phase(1).ipus(1,3)
    strategy.phase(2).ipus(0,2)

    opts.setExecutionStrategy(strategy)


In the code example above, there are three phases. Each phase has two stages
and each IPU group has two IPUs, so the number of groups matches the number
of IPUs. Even phases 0 and 2 run on IPU 0 and 2, while odd phase 1 runs on
IPU 1 and as required. This allows for faster cross-IPU copies, both
inter-phase and intra-phase.

poptorch.Liveness
"""""""""""""""""

:py:class:`poptorch.Liveness` controls the availability of tensors on IPU,
and is only needed for
:py:class:`poptorch.ParallelPhasedExecution`
and :py:class:`poptorch.SerialPhasedExecution`.

The default :py:class:`poptorch.Liveness` is ``AlwaysLive``.
``OffChipAfterFwd`` and
``OffChipAfterEachPhase`` may be helpful if you run a large model
with a tight memory budget.


Optimizers
==========

Poptorch supports the following optimizers:

#. SGD (see :py:class:`poptorch.optim.SGD`)
#. Adam (see :py:class:`poptorch.optim.Adam`)
#. AdamW (see :py:class:`poptorch.optim.AdamW`)
#. RMSprop (see :py:class:`poptorch.optim.RMSprop`)
#. LAMB (see :py:class:`poptorch.optim.LAMB`)

In addition, PopTorch has features to support float16 models, such as loss scaling, velocity scaling, bias correction and accumulator types.

.. important:: All of these extra attributes (Except ``velocity_scaling``) cannot have different values for different ``param_groups`` and therefore must be set at the optimizer level.

.. literalinclude:: api.py
    :language: python
    :caption: How to update values in an Optimizer
    :linenos:
    :start-after: optim_start
    :end-before: optim_end
    :emphasize-lines: 8-12

.. important:: You must call :py:func:`~poptorch.PoplarExecutor.setOptimizer` for the new optimizer values to be applied to the model.

Loss scaling
------------

When training models which use half/float16 values, you can use loss scaling  to prevent the gradients from becoming too small and underflowing.

Before calculating the gradients, PopTorch will scale the loss by the value of the ``loss_scaling`` parameter.
PopTorch will multiply the gradients by the inverse scale prior to updating the optimizer state.
Therefore, beyond improving numerical stability, neither the training nor the hyper-parameters are affected.

Higher ``loss_scaling`` values can improve numerical stability by minimising underflow.
However, too high a value can result in overflow.
The optimal loss scaling factor depends on the model.


Velocity scaling (SGD only)
---------------------------

The SGD optimizer, when used with momentum, updates weights based on the velocity values.
At each update step, the new velocity is a combination of the gradients derived from the loss function and the previous velocity value.
Similar to loss scaling, the ``velocity_scaling`` parameter allows the velocity values to be scaled to improve numerical precision when using half/float16 values.
(Note that the gradients are, in effect, scaled by ``velocity_scaling/loss_scaling`` so the ``loss_scaling`` has no impact on the effective scaling of velocity parameters.)

As with loss scaling, higher values can minimise underflow of the velocity values but may result in overflow.

Accumulation types
------------------

In order to improve numerical stability some of the optimizers (LAMB, Adam, AdamW, RMSprop) give you the option
to tweak the data type used by the optimizer's accumulators.

``accum_type`` lets you choose the type used for gradient accumulation.
``first_order_momentum_accum_type`` / ``second_order_momentum_accum_type`` give you control over the type used to store the first-order and second-order momentum optimizer states.

Constant attributes
-------------------

In order to improve performance and / or save memory PopTorch will try to embed directly in the program the attributes which are constant.

.. important:: Trying to modify a constant attribute after the model has been compiled will result in an error.

For PopTorch optimizers (those from the ``poptorch.optim`` namespace) by default the attributes explicitly passed to the Optimizer's constructor will be considered variables and the others will be considered as constant.

This behaviour can be overridden using :py:func:`~poptorch.optim.VariableAttributes.markAsConstant` and :py:func:`~poptorch.optim.VariableAttributes.markAsVariable` before the model is compiled.

.. literalinclude:: api.py
    :language: python
    :caption: Constant and variable attributes for PopTorch optimizers
    :linenos:
    :start-after: optim_const_start
    :end-before: optim_const_end

For native optimizers (those from the ``torch.optim`` namespace) the attributes which are left to their default value in the constructor will be considered as constant.

There is no method to override this behaviour which is why we recommend you always use the ``poptorch.optim`` optimizers instead.

.. literalinclude:: api.py
    :language: python
    :caption: Constant and variable attributes for Torch optimizers
    :linenos:
    :start-after: torch_optim_const_start
    :end-before: torch_optim_const_end

.. note:: There is an exception: ``lr`` is always marked as variable.

Custom ops
==========

These are helper operations to be used within a model.

poptorch.ipu_print_tensor
-------------------------

Adds an op to print the content of a given IPU tensor.

.. warning::
   To prevent the print operation being optimised out by the graph
   optimiser, you must use the output of the print.

.. literalinclude:: api.py
    :language: python
    :linenos:
    :start-after: print_tensor_start
    :end-before: print_tensor_end
    :emphasize-lines: 10

For more information see: :py:func:`poptorch.ipu_print_tensor`.

poptorch.identity_loss
----------------------

This function is used to implement custom losses. This takes in a single PyTorch tensor
and will backpropagate a gradient of ones through it.

.. warning::
   Passing a PyTorch loss function or another ``identity_loss`` to this function is not
   supported. Multiple losses must be implemented via composite PyTorch ops.


.. literalinclude:: api.py
  :language: python
  :linenos:
  :start-after: identity_start
  :end-before: identity_end
  :emphasize-lines: 5
  :caption: Example of custom loss.

For more information see: :py:func:`poptorch.identity_loss`.

poptorch.MultiConv
------------------

Use :py:class:`poptorch.MultiConv` wrapper class to define multi-convolutions.

Please refer to the `PopLibs documentation for multi-convolutions <https://docs.graphcore.ai/projects/poplar-api/en/latest/poplibs_api.html>`_ for further information.

For more information see: :py:class:`poptorch.MultiConv` :py:class:`poptorch.MultiConvPlanType`.

poptorch.custom_op
------------------

This is for the users who are familiar with PopART.
If you need some special features that are not
supported in PopART, you may write a PopART custom op.
For more information about
how to create Popart custom ops see
`Creating custom operations
<https://docs.graphcore.ai/projects/popart-user-guide/en/latest/custom_ops.html>`_
and
`Building custom operators using PopART
<https://github.com/graphcore/examples/tree/master/code_examples/popart/custom_operators>`_.
You can call such a PopART custom op using
:py:class:`poptorch.custom_op`
in PopTorch.

It takes three steps to enable a PopART custom op in PopTorch.

First, set Poplar and PopART environment varibles as shown in
:ref:`setting_env` and compile the PopART custom op.
You can compile your custom op C++ code and link with Poplar and PopART to
generate a dynamic library.
Please refer to the custom op code custom_cube_op.cpp
and its CMakeLists.txt under
poptorch/tests/custom_ops$.

Second, load the dynamic library.

.. literalinclude:: ../../tests/custom_ops_test.py
    :language: python
    :caption: Loading the library for the PopART custom op
    :linenos:
    :start-after: loading_library_start
    :end-before: loading_library_end

Finally, use :py:class:`poptorch.custom_op` to finish the call.
Its wrapper class is specified below.

For more information see: :py:class:`poptorch.custom_op`.

In the PopART custom op, both forward op and backward op are implemented.
In the PopTorch inference model, only the forward op will be called.

.. literalinclude:: ../../tests/custom_ops_test.py
    :language: python
    :caption: Calling a PopART custom op in a PopTorch inference model
    :linenos:
    :emphasize-lines: 4-8
    :start-after: inference_start
    :end-before: inference_end

In the code example above, ``example_outputs`` is assigned as
[``x``, ``x``], where ``x`` is one of the input tensors and used as
a template to provide the right number of output tensors.
The real outputs will be allocated memory, calculated and
returned by the custom op.
You can also call this custom op inside a training model
using exactly the same interface of :py:class:`poptorch.custom_op`,
and the backward op will be called automatically.

You can pass attributes to custom ops using a Python dictionary, as shown by the
following code example:

.. literalinclude:: ../../tests/custom_ops_test.py
    :language: python
    :caption: Passing an attribute to a PopART custom op from PopTorch
    :linenos:
    :emphasize-lines: 8
    :start-after: inference_with_attribute_start
    :end-before: inference_with_attribute_end

You can then obtain attributes from within the C++ code. The above code
passes a ``Float`` attribute with the name ``alpha`` to the LeakyRELU implementation in the `Custom operations chapter of the PopART user guide
<https://docs.graphcore.ai/projects/popart-user-guide/en/latest/custom_ops.html>`_.
PopTorch supports all possible attributes supported in PopArt except for
``Graph``.

Please refer to the following table and code examples for information on how to
pass other attribute types to a PopArt custom op implementation:

.. list-table:: Python types to use to pass attributes to PopART
   :widths: 35 65
   :header-rows: 1

   * - PopART attribute type
     - Python equivalent
   * - ``Float``
     - Python float (converted to 32-bit)
   * - ``Floats``
     - list/tuple of Python floats
   * - ``Int``
     - Python int (converted to 64-bit signed int)
   * - ``Ints``
     - list/tuple of Python ints
   * - ``String``
     - Python str (converted to ASCII)
   * - ``Strings``
     - List/tuple of Python strs
   * - ``Graph``
     - Not supported

.. literalinclude:: ../../tests/custom_ops_attributes_test.py
    :language: python
    :caption: Passing different attribute types from PopTorch
    :linenos:
    :start-after: many_attribtes_examples_start
    :end-before: many_attribtes_examples_end

poptorch.nop
------------

Poptorch includes a "no-op" function for debugging purposes.

For more information see: :py:func:`poptorch.nop`.

poptorch.serializedMatMul
-------------------------

Use this function to create a serialized matrix multiplication, which splits
a larger matrix multiplication into smaller matrix multiplications to reduce
memory requirements.

For more information see: :py:func:`poptorch.serializedMatMul`.


poptorch.set_available_memory
-----------------------------

Use this function to override the proportion of tile memory for available to be used as temporary memory by a convolution or matrix multiplication.

For more information see: :py:func:`poptorch.set_available_memory`.

Miscellaneous functions
=======================

These PopTorch functions, not related to model creation, are available:

- :py:func:`poptorch.ipuHardwareIsAvailable`
- :py:func:`poptorch.ipuHardwareVersion`
- :py:func:`poptorch.setLogLevel`


Half / float 16 support
=======================

PopTorch supports the half-precision floating point (float 16) format.
You can simply input float 16 tensors into your model.
(You can convert a tensor to float 16 using ``tensor = tensor.half()``)

You can use your models in one of the following ways:

#. Convert all parameters (weights) to float 16 by using using a ``Module``'s .``half()`` method. This is the most memory efficient, however small updates to weights may be lost, hindering training.
#. Keep the parameters (weights) as float 32, in which case the parameter updates will occur using float 32. However, the parameters will be converted to float 16 if you call an operation with a float 16 input. This is more memory efficient than using float 32 tensors (inputs) but less memory efficient than using float 16 weights.
#. Use a mix of float 32 and float 16 parameters by manually specifying parameters as float 16 or float 32.

.. note::  When PyTorch encounters a mix of float 16 and float 32 inputs for a given operation, it will usually cast all inputs and float 32.
    PopTorch differs and will cast all inputs to float 16.
    This makes it easier to build models with float 32 weights which take float 16 tensors. However, if you wish to follow PyTorch behavior, you can use  ``opts.Precision.halfFloatCasting(poptorch.HalfFloatCastingBehavior.HalfUpcastToFloat)`` where ``opts`` is the ``poptorch.Options`` object passed to the model wrapping function.

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
Some operations may result in the model running in float 32 where float 16 would
be expected, or vice versa (see :ref:`float_16_op_support` for full details).

Profiling
=========

You can profile a graph produced by PopTorch for analysis using the PopVision Graph Analyser, which can be downloaded from the Graphcore support portal.
To do this, use the :ref:`POPLAR_ENGINE_OPTIONS<profiling_env>` environment variable.

Precompilation and caching
==========================

.. _caching :

Caching
-------

By default PopTorch will re-compile the model every time you instantiate a model.
However if you often run the same models you might want to enable executable caching to save time.

You can do this by either setting the ``POPTORCH_CACHE_DIR`` environment variable or by calling :py:class:`poptorch.Options.enableExecutableCaching`.

.. warning:: The cache directory might grow large quickly because PopTorch doesn't evict old models from the cache and, depending on the number and size of your models and the number of IPUs used, the executables might be quite large. It is the your responsibility to delete the unwanted cache files.

Precompilation
--------------

PopTorch supports precompilation: This means you can compile your model on a machine which doesn't have an IPU
and export the executable to a file. You can then reload and execute it on a different machine which does have an IPU.

.. important:: The PopTorch versions on both machines must be an exact match.

To precompile your model you need to wrap it using either :py:func:`poptorch.trainingModel` or :py:func:`poptorch.inferenceModel` then call :py:meth:`~poptorch.PoplarExecutor.compileAndExport` on the wrapper.

.. literalinclude:: precompilation.py
    :language: python
    :caption: How to precompile a model using an offline IPU target.
    :linenos:
    :start-after: precomp_start
    :end-before: precomp_end
    :emphasize-lines: 22-23,32

.. note:: If you don't know the IPU version on your system you can use :py:func:`poptorch.ipuHardwareVersion`.

The exported file by default will contain your original Torch model (including the weights), and enough information to re-create the PopTorch wrapper and reload the executable.

.. important:: For your model and weights to be exported, your model must be picklable. See https://docs.python.org/3/library/pickle.html for more information.
  If your model is not picklable then use ``export_model=False``, see :ref:`below for a complete example<export_no_python>`.

Now both the torch model, PopTorch wrapper and executable can be restored on the target machine using :py:func:`poptorch.load`:

.. literalinclude:: precompilation.py
    :language: python
    :caption: How to load a precompiled model.
    :linenos:
    :start-after: load_start
    :end-before: load_end
    :emphasize-lines: 1

In some cases you might want to provide some runtime information to select the device: this can be done
using the `edit_opts_fn` argument of :py:func:`poptorch.load`:

.. literalinclude:: precompilation.py
    :language: python
    :caption: How to load a precompiled model and run on a specific IPU.
    :linenos:
    :start-after: load_setIpu_start
    :end-before: load_setIpu_end
    :emphasize-lines: 1-2,5

.. note:: Only runtime options will be used as the executable is already compiled

.. _export_no_python:

Going back to the precompilation step: in some cases you might want to export only the
executable and not the python wrapper or torch model (For example if your model cannot be pickled).

.. literalinclude:: precompilation.py
    :language: python
    :caption: How to export only the exectuable.
    :linenos:
    :start-after: precomp_no_python_start
    :end-before: precomp_no_python_end

It means you will need to re-create and wrap the model yourself before loading the executable:

.. literalinclude:: precompilation.py
    :language: python
    :caption: How to load a precompiled executable.
    :linenos:
    :start-after: load_exe_start
    :end-before: load_exe_end
    :emphasize-lines: 1,6-7

.. important:: Exported models lose their connections to other models.

For example, if you have a :py:func:`poptorch.trainingModel` and a :py:func:`poptorch.inferenceModel` based
on the same PyTorch model, you wouldn't usually need to keep the weights synchronised between the two:
PopTorch would take care of it implicitly for you.

For example:

.. literalinclude:: precompilation.py
    :language: python
    :caption: PopTorch implicit copies.
    :linenos:
    :start-after: implicit_cp_start
    :end-before: implicit_cp_end
    :emphasize-lines: 16,18-20

If you were to export these models:

.. literalinclude:: precompilation.py
    :language: python
    :caption: Precompilation of both a training and validation models.
    :linenos:
    :start-after: precomp_train_val_start
    :end-before: precomp_train_val_end
    :emphasize-lines: 11-12,14

.. note:: Don't forget to ``model.eval()`` or ``model.train()`` as required before calling :py:func:`~poptorch.PoplarExecutor.compileAndExport`.

You would then either need to insert explicit copy operations:

.. literalinclude:: precompilation.py
    :language: python
    :caption: Precompilation of both a training and validation models.
    :linenos:
    :start-after: load_train_val_start
    :end-before: load_train_val_end
    :emphasize-lines: 9,10

Or you would need to re-connect the two models by creating the second one from the first one
and then loading the executable:

.. literalinclude:: precompilation.py
    :language: python
    :caption: Precompilation of both a training and validation models.
    :linenos:
    :start-after: load_train_val_connected_start
    :end-before: load_train_val_connected_end
    :emphasize-lines: 2-6


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

In order to capture the ``pvti`` reports needed for the PopVision System Analyser you only need to set ``PVTI_OPTIONS='{"enable":"true"}'``

You can also add extra tracepoints in your own code by using :py:class:`poptorch.profiling.Channel`.

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

Enable executable caching
-------------------------

This can be done by either setting the ``POPTORCH_CACHE_DIR`` environment variable or by calling :py:class:`poptorch.Options.enableExecutableCaching`.

.. seealso:: For more info :ref:`caching<caching>`

.. code-block:: bash

  export POPTORCH_CACHE_DIR=/tmp/poptorch_cache

