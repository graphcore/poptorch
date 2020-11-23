# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import enum


class OptimizerType(enum.IntEnum):
    SGD = 0
    ADAMW = 1
    RMSPROP = 2
    RMSPROP_CENTERED = 3
    LAMB = 4
    LAMB_NO_BIAS = 5


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
    - ``OffChipAfterEachPhase``: The tensors are sent off the chip at the end
      of each phase.
    """
    AlwaysLive = 0
    OffChipAfterFwd = 1
    OffChipAfterEachPhase = 2


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
