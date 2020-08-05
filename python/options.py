# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from . import _impl
from . import enums
from .logging import logger


class _JitOptions(_impl.OptionsDict):
    """Options related to Pytorch's JIT
    """

    def __init__(self):
        super().__init__(trace_model=True)

    def traceModel(self, trace_model):
        """
        If True: use torch.jit.trace
        If False: use torch.jit.script

        Trace model is enabled by default.
        """
        self.set(trace_model=trace_model)
        return self


class _TrainingOptions(_impl.OptionsDict):
    """Options specific to model training.
    """

    def __init__(self):
        super().__init__(gradient_accumulation=1)

    def gradientAccumulation(self, gradient_accumulation):
        self.set(gradient_accumulation=gradient_accumulation)
        return self


class _PopartOptions:
    """Options specific to the Popart backend.
    Only for advanced users.
    """

    def __init__(self):
        self.options = {}

    def set(self, key, value):
        self.options[key] = value
        return self


class Options(_impl.OptionsDict):
    def __init__(self):
        self._jit = _JitOptions()
        self._training = _TrainingOptions()
        self._popart = _PopartOptions()

        super().__init__(
            replication_factor=1,
            device_iterations=1,
            log_dir=".",
            profile=False,
            anchor_mode=enums.AnchorMode.Default.value,
            anchor_return_period=1,
            use_model=False,
            connection_type=enums.ConnectionType.Always.value,
            sync_pattern=enums.SyncPattern.Full.value,
        )

    @property
    def Jit(self):
        """Options specific to PyTorch's JIT."""
        return self._jit

    @property
    def Training(self):
        """Options specific to training."""
        return self._training

    @property
    def Popart(self):
        """Options specific to the Popart backend.
        (Advanced users only).
        """
        return self._popart

    def deviceIterations(self, device_iterations):
        """Number of iterations run on the device per execution (Default: 1)"""
        self.set(device_iterations=device_iterations)
        return self

    def enablePipelining(self, enable_pipelining):
        """Enable pipelining of virtual graphs (Default: False if 1 IPU used,
        True otherwise)"""
        self.createOrSet(enable_pipelining=enable_pipelining)
        return self

    def replicationFactor(self, replication_factor):
        """Number of model replications (Default: 1).

        E.g. if your model uses 1 IPU, a
        replication factor of 2 will use 2 IPUs. If your model is
        pipelined across 4 IPUs, a replication factor of 4 will use 16 IPUs
        total.
        """
        self.set(replication_factor=replication_factor)
        return self

    def logDir(self, log_dir):
        """Where to save log files (Default: Current directory)"""
        self.set(log_dir=log_dir)
        return self

    def profile(self, profile):
        """Enable profiling (Default: False)"""
        self.set(profile=profile)
        return self

    def useIpuModel(self, use_model):
        """Use the IPU model or physical hardware.

        Default: False (Real Hardware)
        This setting takes precedence over the POPTORCH_IPU_MODEL environment
        variable.
        """
        self.set(use_model=use_model)
        return self

    def connectionType(self, connection_type):
        """set the IPU connection type to one of:
        - Always: Attach to the IPU from the start (Default).
        - OnDemand: Wait until the compilation is complete and the executable
          is ready to be run to attach to the IPU.
        - Never: Never try to attach to an IPU. (Useful for offline compilation,
          but trying to run an executable will raise an exception).
        """
        assert isinstance(connection_type, enums.ConnectionType)
        self.set(connection_type=connection_type.value)
        return self

    def syncPattern(self, sync_pattern):
        """set the IPU SyncPatter to one of:
        - Full
        - SinglePipeline
        - PingPong
        """
        assert isinstance(sync_pattern, enums.SyncPattern)
        self.set(sync_pattern=sync_pattern.value)
        return self

    def useIpuId(self, ipu_id):
        """ Use the specified IPU id as provided by gc-info.

        The number of IPUs associated with the id must be equal to the number
        of IPUs used by your grpah multiplied by the replication factor.

        E.g. if your model uses 1 IPU and the replication factor is 2 you will
        need to provide an id with 2 IPUs.
        If your model is pipelined across 4 IPUs, the replication factor is 4,
        you will need to provide an id containing 16 IPUs total.
        """
        assert isinstance(ipu_id, int)
        self.createOrSet(ipu_id=ipu_id)
        return self

    def useOfflineIpuTarget(self, ipu_version=1):
        """Create an offline IPU target that can only be used for offline compilation.

        Note: the offline IPU target cannot be used if the IPU model is enabled.
        """
        self.connectionType(enums.ConnectionType.Never)
        self.createOrSet(ipu_version=ipu_version)
        return self

    def anchorMode(self, anchor_mode, anchor_return_period=None):
        """ How much data to return from a model

        Args:
            anchor_mode:
                All: Return a result for each batch.
                Sum: Return the sum of all the batches
                Final: Return the last batch.
                EveryN: Return every N batches. N is passed in as
                    |anchor_return_period|
                Default: "All" for inference, "Final" for training.
        """
        assert isinstance(anchor_mode, enums.AnchorMode)

        # Check the anchor return period makes sense.
        if anchor_mode == enums.AnchorMode.EveryN:
            assert anchor_return_period and anchor_return_period > 0, (
                "EveryN"
                " anchor must have anchor_return_period set to valid"
                " positive integer")
        elif anchor_return_period:
            logger.info(
                "Anchor return period argument ignored with anchor_mode"
                " set to %s", anchor_mode)

        self.set(anchor_mode=anchor_mode.value,
                 anchor_return_period=anchor_return_period or 1)
        return self

    def defaultAnchorMode(self):
        """Return True if the anchor_mode is currently set to Default,
        False otherwise."""
        return self.anchor_mode == enums.AnchorMode.Default

    def toDict(self):
        """ Merge all the options, except for the Jit ones, into a ringle
        dictionary to be serialised and passed to the cpp side."""
        assert not self.defaultAnchorMode(
        ), "An anchor mode must be picked before serialisation"
        out = {}
        out.update(self._popart.options)
        out = self.update(out)
        out = self._training.update(out)
        return out
