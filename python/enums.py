# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import enum


class DataLoaderMode(enum.IntEnum):
    """
    - ``Sync``: Access data synchronously
    - ``Async``: Uses an :py:class:`~poptorch.AsynchronousDataAccessor`
      to access the dataset
    """
    Sync = 0
    Async = 1


class SharingStrategy(enum.IntEnum):
    """Strategy to use to pass objects when spawning new processes.

    - ``SharedMemory``: Fast but limited availability.
    - ``FileSystem``: Slower but larger than memory.
    """
    SharedMemory = 0
    FileSystem = 1


class AnchorMode(enum.IntEnum):
    """
    - ``All``: Return a result for each batch.
    - ``Sum``: Return the sum of all the batches
    - ``Final``: Return the last batch.
    - ``EveryN``: Return every N batches. N is passed in as
        `anchor_return_period`
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
    - ``FloatDowncastToHalf`` Any op with operands (inputs) which are a
        mix of float32 and float16 (half) will cast all operands to half.
    - ``HalfUpcastToFloat``: Implicit casting will follow Pytorch's rules,
            promoting float16 (half) inputs to float32 if another input is
            float32.
    """
    FloatDowncastToHalf = 0
    HalfUpcastToFloat = 1


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
    - ``Full``
    - ``SinglePipeline``
    - ``ReplicaAndLadder``
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

    >>> stategy = poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu)
    >>> opts.setExecutionStrategy(strategy)

    which would translate to ``stage_id = ipu_id``:

    - Block "0" ipu=0 stage=0
    - Block "1" ipu=1 stage=1
    - Block "2" ipu=0 stage=0

    Now if instead you use:

    >>> stategy = poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement)
    >>> opts.setExecutionStrategy(strategy)

    The last block would be in its own stage rather than sharing one with
    Block "0":

    - Block "0" ipu=0 stage=0
    - Block "1" ipu=1 stage=1
    - Block "2" ipu=0 stage=2
    """
    SameAsIpu = 0
    AutoIncrement = 1


# TODO(T34238): enums.MultiConvPartialsType deprecated in 2.0
class MultiConvPartialsType(enum.IntEnum):
    """Type for the partials of each convolution of a ``poptorch.MultiConv``

    - ``Float``
    - ``Half``
    """
    Float = 0
    Half = 1


class MultiConvPlanType(enum.IntEnum):
    """Selects the execution strategy for a ``poptorch.MultiConv``

    - ``Parallel``: Execute multiple convolutions in parallel (Default).
    - ``Serial``: Execute each convolution independently. This is
      equivalent to using the independent convolution API.
    """
    Parallel = 0
    Serial = 1
