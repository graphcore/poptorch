.. _reference:

=============
API reference
=============

.. _api_options:

Options
=======

.. autoclass:: poptorch.Options
   :members:

.. autoclass:: poptorch.options._DistributedOptions
   :members:

.. autoclass:: poptorch.options._PrecisionOptions
   :members:

.. autoclass:: poptorch.options._JitOptions
   :members:

.. autoclass:: poptorch.options._TensorLocationOptions
   :members:

.. autoclass:: poptorch.TensorLocationSettings
   :members:

.. autoclass:: poptorch.options._TrainingOptions
   :members:

Helpers
=======

.. autofunction:: poptorch.ipuHardwareIsAvailable

.. autofunction:: poptorch.ipuHardwareVersion

.. autofunction:: poptorch.setLogLevel

.. autoclass:: poptorch.profiling.Channel
   :members:

PopTorch ops
============

.. autofunction:: poptorch.ctc_beam_search_decoder

.. autofunction:: poptorch.ipu_print_tensor

.. autofunction:: poptorch.for_loop

.. autofunction:: poptorch.recomputationCheckpoint

.. autofunction:: poptorch.identity_loss

.. autoclass:: poptorch.MultiConv
   :members:

.. autoclass:: poptorch.CPU
   :special-members: __init__

.. autoclass:: poptorch.NameScope
   :members:

.. autoclass:: poptorch.MultiConvPlanType

.. autoclass:: poptorch.custom_op

.. autofunction:: poptorch.nop

.. autofunction:: poptorch.dynamic_slice

.. autofunction:: poptorch.dynamic_update

.. autofunction:: poptorch.serializedMatMul

.. autofunction:: poptorch.set_available_memory

.. autofunction:: poptorch.set_overlap_for_input

.. autofunction:: poptorch.set_overlap_for_output

.. autofunction:: poptorch.nearest

.. autofunction:: poptorch.fps

.. autofunction:: poptorch.cond

Model wrapping functions
========================

.. autofunction:: poptorch.trainingModel

.. autofunction:: poptorch.inferenceModel

.. autoclass:: poptorch.PoplarExecutor
   :special-members: __call__
   :members:

.. autofunction:: poptorch.isRunningOnIpu

.. autofunction:: poptorch.load

Parallel execution
==================

.. autoclass:: poptorch.Block
   :special-members: __init__, useAutoId

.. autoclass:: poptorch.BeginBlock
   :special-members: __init__

.. autofunction:: poptorch.BlockFunction

.. autofunction:: poptorch.removeBlocks

.. autoclass:: poptorch.Stage
   :special-members: __init__

.. autoclass:: poptorch.AutoStage

.. autoclass:: poptorch.Phase
   :special-members: __init__

.. autoclass:: poptorch.ShardedExecution
   :inherited-members:

.. autoclass:: poptorch.PipelinedExecution
   :special-members: __init__
   :inherited-members:

.. autoclass:: poptorch.SerialPhasedExecution
   :special-members: __init__
   :inherited-members:

.. autoclass:: poptorch.ParallelPhasedExecution
   :special-members: __init__
   :inherited-members:

.. autoclass:: poptorch.Liveness

.. autoclass:: poptorch.CommGroupType

.. autoclass:: poptorch.VariableRetrievalMode

.. py:function:: replicaGrouping

   Call this function on a weight tensor (after applying a PopTorch wrapper with
   :py:func:`~poptorch.inferenceModel` or :py:func:`~poptorch.trainingModel`)
   to configure replica groups which each receive a different value of the
   weight tensor. For details and a code example see
   :numref:`grouping_tensor_weights`.

   :param comm_group_type: The replica group arrangement to use for this tensor.
   :type comm_group_type: poptorch.CommGroupType
   :param shards: The number of replicas in each replica group.
   :type shards: int
   :param variable_retrieval_mode: The method to use when retrieving the value
                                   of this tensor from the replicas.
   :type variable_retrieval_mode: poptorch.VariableRetrievalMode

Optimizers
==========

.. autoclass:: poptorch.optim.VariableAttributes
   :members:

.. autoclass:: poptorch.optim.SGD
   :special-members: __init__
   :members:

.. autoclass:: poptorch.optim.Adam
   :special-members: __init__
   :members:

.. autoclass:: poptorch.optim.AdamW
   :special-members: __init__
   :members:

.. autoclass:: poptorch.optim.RMSprop
   :special-members: __init__
   :members:

.. autoclass:: poptorch.optim.LAMB
   :special-members: __init__
   :members:

Data batching
=============

.. autoclass:: poptorch.DataLoader
   :special-members: __init__
   :members: terminate

.. autoclass:: poptorch.AsynchronousDataAccessor
   :special-members: __init__, __len__
   :members: terminate

.. autoclass:: poptorch.DataLoaderMode
   :members:

Enumerations
============

.. autoclass:: poptorch.SharingStrategy
   :members:

.. autoclass:: poptorch.OverlapMode
   :members:

.. autoclass:: poptorch.MatMulSerializationMode
   :members:

.. autoclass:: poptorch.SyncPattern
   :members:

.. autoclass:: poptorch.ReductionType
   :members:

.. autoclass:: poptorch.ConnectionType
   :members:

.. autoclass:: poptorch.OutputMode
   :members:

.. autoclass:: poptorch.MeanReductionStrategy
   :members:
