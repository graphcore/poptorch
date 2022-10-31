# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import enum


class MeanReductionStrategy(enum.IntEnum):
    """Specify when to divide by a mean reduction factor when
    ``accumulationAndReplicationReductionType`` is set to
    ``ReductionType.Mean``.

    - ``Running``: Keeps the reduction buffer as the current mean. This is
      preferred for numerical stability as the buffer value is never larger than
      the magnitude of the largest micro batch gradient.
    - ``Post``: Divides by the accumulationFactor and replicatedGraphCount after
      all of the gradients have been reduced. In some cases this can be
      faster then using Running, however is prone to overflow.
    - ``PostAndLoss`` (deprecated): Divides by the replicatedGraphCount before
      the backwards pass, performs the gradient reduction across micro batches,
      and then divides by the accumulationFactor. This is to support legacy
      behaviour and is deprecated.
    """
    Running = 0
    Post = 1
    PostAndLoss = 2


class DataLoaderMode(enum.IntEnum):
    """
    - ``Sync``: Access data synchronously
    - ``Async``: Uses an :py:class:`~poptorch.AsynchronousDataAccessor`
      to access the dataset
    - ``AsyncRebatched``: For iterable datasets by default PyTorch will round
      down the number of elements to a multiple of the combined batch size in
      each worker. When the number of workers is high and/or the batch size
      large this might lead to a significant part of the dataset being
      discarded. In this mode, the
      combined batch size used by the PyTorch workers will be set to 1,
      and the batched tensor will instead be constructed in the
      :py:class:`~poptorch.AsynchronousDataAccessor`.
      This mode is identical to Async for map-style datasets.
    """
    Sync = 0
    Async = 1
    AsyncRebatched = 2


class SharingStrategy(enum.IntEnum):
    """Strategy to use to pass objects when creating new processes.

    - ``SharedMemory``: Spawn new processes and share data using shared memory:
                        Fast but limited availability.
    - ``FileSystem``: Spawn new processes and shared data using the file
                      system: slower but larger than memory.
    - ``Fork``: Fork new processes: no data sharing required but might cause
                problems if worker processes use threading.
    - ``ForkServer``: Similar to fork but a server process is used to fork child
                      processes instead. This server process is single-threaded
                      so there are no issues if worker processes use threading.
    """
    SharedMemory = 0
    FileSystem = 1
    Fork = 2
    ForkServer = 3


class OutputMode(enum.IntEnum):
    """
    - ``All``: Return a result for each batch.
    - ``Sum``: Return the sum of all the batches
    - ``Final``: Return the last batch.
    - ``EveryN``: Return every N batches. N is passed in as
        `output_return_period`
    - ``Default``: "All" for inference, "Final" for training.
    """
    Final = 0
    EveryN = 1
    All = 2
    Sum = 3
    Default = 4


class ConnectionType(enum.IntEnum):
    """
    - ``Always``: Attach to the IPU from the start (Default).
    - ``OnDemand``: Wait until the compilation is complete and the executable is
      ready to be run to attach to the IPU.
    - ``Never``: Never try to attach to an IPU. (Useful for offline compilation,
      but trying to run an executable will raise an exception).
    """
    Always = 0
    OnDemand = 1
    Never = 2


class HalfFloatCastingBehavior(enum.IntEnum):
    """
    - ``FloatDowncastToHalf``: Any op with operands (inputs) which are a
        mix of float32 and float16 (half) will cast all operands to half.
    - ``HalfUpcastToFloat``: Implicit casting will follow PyTorch's rules,
        promoting float16 (half) inputs to float32 if another input is float32.
    - ``Default``: This is ``FloatDowncastToHalf`` for tracing, and
        ``HalfUpcastToFloat`` for the dispatcher, which only supports following
        PyTorch's casting rules.
    """
    FloatDowncastToHalf = 0
    HalfUpcastToFloat = 1
    Default = 2


class ReductionType(enum.IntEnum):
    """
    - ``Sum``: Calculate the sum of all values
    - ``Mean``: Calculate the mean of all values
    - ``NoReduction``: Do not reduce
    """
    Sum = 0
    Mean = 1
    NoReduction = 2


class SyncPattern(enum.IntEnum):
    """
    - ``Full``: Require all IPUs to synchronise on every communication between
      IPUs or between IPUs and host.
    - ``SinglePipeline``: Allow IPUs to synchronise with the host independently,
      without having to synchronise with each other. This permits any one IPU to
      perform host IO while other IPUs are processing data.
    - ``ReplicaAndLadder``: Allow an IPU group to communicate with the host
      without requiring synchronisation between groups. This permits multiple
      IPU groups to alternate between performing host IO and computation.
    """
    Full = 0
    SinglePipeline = 1
    ReplicaAndLadder = 2


class MatMulSerializationMode(enum.Enum):
    """Which dimension of the matrix multiplication to use for the
    serialization"""
    InputChannels = "input_channels"
    ReducingDim = "reducing_dim"
    OutputChannels = "output_channels"
    Disabled = "none"


class Liveness(enum.IntEnum):
    """When using phased execution:

    - ``AlwaysLive``: The tensors always stay on the IPU between the phases.
    - ``OffChipAfterFwd``: The tensors are sent off the chip at the end of
      the forward pass and before the beginning of the backward pass.
    - ``OffChipAfterFwdNoOverlap``: Same as `OffChipAfterFwd`, except there is
      no overlapping of load and store operations between phases. This makes it
      a more memory-efficient mode at the cost of delayed computation.
    - ``OffChipAfterEachPhase``: The tensors are sent off the chip at the end
      of each phase.
    """
    AlwaysLive = 0
    OffChipAfterFwd = 1
    OffChipAfterFwdNoOverlap = 2
    OffChipAfterEachPhase = 3


class OverlapMode(enum.Enum):
    """
    - ``NoOverlap``: The host will copy the tensor to the IPU only when
      required: this minimises on-chip memory use at the cost of performance.
    - ``OverlapAccumulationLoop``: The host will preload values for the next
      gradient accumulation iteration onto an IO tile.
    - ``OverlapDeviceIterationLoop``: The host will preload values not just for
      the next gradient accumulation iteration, but the next device iteration,
      onto an IO tile. This may require more IO tiles than the previous setting
      but offers greater performance.
    - """
    NoOverlap = "no_overlap"
    OverlapAccumulationLoop = "overlap_accumulation_loop"
    OverlapDeviceIterationLoop = "overlap_device_iteration_loop"


class AutoStage(enum.IntEnum):
    """Defines how the stages are automatically assigned to blocks when the user
    didn't explicitly provide stages to the ``IExecutionStrategy``'s
    constructor.

    - ``SameAsIpu``: The stage id will be set to the selected ipu number.
    - ``AutoIncrement``: The stage id for new blocks is automatically
      incremented.

    Examples:

    >>> # Block "0"
    >>> with poptorch.Block(ipu_id=0):
    ...  layer()
    >>> # Block "1"
    >>> with poptorch.Block(ipu_id=1):
    ...  layer()
    >>> # Block "2"
    >>> with poptorch.Block(ipu_id=0):
    ...  layer()

    By default, the following execution strategy is used:

    >>> strategy = poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu)
    >>> opts.setExecutionStrategy(strategy)

    which would translate to ``stage_id = ipu_id``:

    - Block "0" ipu=0 stage=0
    - Block "1" ipu=1 stage=1
    - Block "2" ipu=0 stage=0

    Now if instead you use:

    >>> strategy = poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement)
    >>> opts.setExecutionStrategy(strategy)

    The last block would be in its own stage rather than sharing one with
    Block "0":

    - Block "0" ipu=0 stage=0
    - Block "1" ipu=1 stage=1
    - Block "2" ipu=0 stage=2
    """
    SameAsIpu = 0
    AutoIncrement = 1


class MultiConvPlanType(enum.IntEnum):
    """Selects the execution strategy for a ``poptorch.MultiConv``

    - ``Parallel``: Execute multiple convolutions in parallel (Default).
    - ``Serial``: Execute each convolution independently. This is
      equivalent to using the independent convolution API.
    """
    Parallel = 0
    Serial = 1


class CommGroupType(enum.IntEnum):
    """Grouping to be used when distributing an input or per-replica variable
       among replicas. See :ref:`grouping_tensor_weights`.

    - ``All``: This causes :py:func:`~replicaGrouping` to have no effect, as the
               same variable value is distributed to all replicas. Group count
               is ignored. This is not valid as an input group type.

    - ``Consecutive``: Each replica group is made up of consecutive replicas,
                       So for group size ``k``, the groups would be set up thus:

                       ``{0, 1, ... k-1}, {k, ... 2k-1} ... {N-k-1, ... N-1}``

    - ``Orthogonal``: Each replica group is made up by slicing the replicas
                      orthogonally to the replica ordering. So for group size
                      ``k``, with group count ``m = N/k``:

                      ``{0, m, 2m, ...}, {1, m+1, 2m+1, ...} ... {m-1, 2m-1,
                      ... N-1}``

    - ``NoGrouping``: Each replica gets its own value of the variable. Group
                      count is ignored.
    """
    All = 0
    Consecutive = 1
    Orthogonal = 2
    NoGrouping = 3


class VariableRetrievalMode(enum.IntEnum):
    """Method to be used when retrieving the value of a grouped variable from
       grouped replicas. See :ref:`grouping_tensor_weights`.

    - ``OnePerGroup``: Return one value for each replica group (takes the value
                       from the first replica in the group).

    - ``AllReplicas``: Return a value from each replica.
    """
    OnePerGroup = 0
    AllReplicas = 2
