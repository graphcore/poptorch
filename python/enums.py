# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import enum


class OptimizerType(enum.IntEnum):
    SGD = 0
    ADAM = 1


class AnchorMode(enum.IntEnum):
    """
    All: Return a result for each batch.
    Sum: Return the sum of all the batches
    Final: Return the last batch.
    EveryN: Return every N batches. N is passed in as |anchor_return_period|
    Default: "All" for inference, "Final" for training.
    """
    Final = 0
    EveryN = 1
    All = 2
    Sum = 3
    Default = 4


class ConnectionType(enum.IntEnum):
    """
    - Always: Attach to the IPU from the start (Default).
    - OnDemand: Wait until the compilation is complete and the executable is
      ready to be run to attach to the IPU.
    - Never: Never try to attach to an IPU. (Useful for offline compilation,
      but trying to run an executable will raise an exception).
    """
    Always = 0
    OnDemand = 1
    Never = 2


class SyncPattern(enum.IntEnum):
    Full = 0
    SinglePipeline = 1
    ReplicaAndLadder = 2
