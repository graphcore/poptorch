# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import glob
import json
import os
import torch
from . import enums
from .logging import logger


class _OptionsDict:
    """Safe dictionary to store options: only keys which have been passed to
    the constructor can later be updated.
    """

    def __init__(self, **default_values):
        self._values = default_values

    def set(self, **kwargs):
        for option, value in kwargs.items():
            assert self.exists(option), ("Invalid option %s, valid options"
                                         " are %s") % (option,
                                                       self._values.keys())
            assert isinstance(
                value, type(self._values[option])
            ), "Unexpected type %s for option %s. Expected %s" % (
                type(value), option, type(self._values[option]))
            self._values[option] = value

    def createOrSet(self, **kwargs):
        for option, value in kwargs.items():
            if self.exists(option):
                self.set(option=value)
            else:
                self._values[option] = value

    def exists(self, option):
        return option in self._values

    def __getattr__(self, option):
        assert self.exists(
            option), ("Invalid option %s, "
                      "valid options are %s") % (option, self._values.keys())
        return self._values[option]

    def update(self, other):
        assert not set(self._values.keys()).intersection(
            other), "Can't merge dictionaries, they have some keys in common"
        other.update(self._values)
        return other

    def __call__(self, option):
        assert self.exists(
            option), ("Invalid option %s, "
                      "valid options are %s") % (option, self._values.keys())
        return self._values[option]


class _JitOptions(_OptionsDict):
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


class _TrainingOptions(_OptionsDict):
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


class _DistributedOptions(_OptionsDict):
    """Options related to distributed execution.

    Note: hostId and numHosts are set by MPI and therefore are read only.
    To change the global replication factor change the `np` value used to
    invoke your script.
    e.g: mpirun -np 4 myscript.py
    """

    def __init__(self):
        super().__init__(num_distributed_hosts=1,
                         distributed_host_id=0,
                         ipuof_configs={})
        self._gcd_mappings = {}
        self.setEnvVarNames("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK")

    def disable(self):
        self.set(num_distributed_hosts=1, distributed_host_id=0)
        return self

    def IPUoFConfigFiles(self, files):
        """ List of IPUoF configuration files to use for the different
        GCDs

        files: one or more glob compatible expressions

        By default: "~/.ipuof.conf.d/*.conf"
        """
        if isinstance(files, str):
            files = [files]
        # Find all the config files
        all_files = []
        for f in files:
            all_files += glob.glob(os.path.expanduser(f))
        # remove duplicates
        all_files = set(all_files)
        self._gcd_mappings = {}
        for f in all_files:
            id = json.load(open(f))["attributes"].get("GCD Id")
            gcd = int(id) if id else 0
            assert gcd not in self._gcd_mappings, (
                f"Multiple config files "
                f"are registered to handle GCD {gcd}: {self._gcd_mappings[gcd]}"
                f" and {f}")
            self._gcd_mappings[gcd] = f
        return self

    def setEnvVarNames(self, var_num_hosts, var_host_id):
        """Set the environment variables names to use to get the number of hosts
        and the host identifier of the current process.

        By default the OpenMPI "OMPI_COMM_WORLD_SIZE" and "OMPI_COMM_WORLD_RANK"
        variables are used.
        """
        return self.configureProcessId(int(os.environ.get(var_host_id, "0")),
                                       int(os.environ.get(var_num_hosts, "1")))

    def configureProcessId(self, host_id, num_hosts):
        """Manually set the current process ID and the total number of hosts.
        """
        self.set(distributed_host_id=host_id)
        self.set(num_distributed_hosts=num_hosts)
        return self

    def getGcdConfigFile(self):
        if not self._gcd_mappings:
            self.IPUoFConfigFiles("~/.ipuof.conf.d/*.conf")
        return self._gcd_mappings.get(self.hostId)

    @property
    def hostId(self):
        return self.distributed_host_id

    @property
    def numHosts(self):
        return self.num_distributed_hosts


class Options(_OptionsDict):
    def __init__(self):
        self._jit = _JitOptions()
        self._training = _TrainingOptions()
        self._popart = _PopartOptions()
        self._distributed = _DistributedOptions()

        super().__init__(
            replication_factor=1,
            device_iterations=1,
            log_dir=".",
            anchor_mode=enums.AnchorMode.Default.value,
            anchor_return_period=1,
            use_model=False,
            connection_type=enums.ConnectionType.Always.value,
            sync_pattern=enums.SyncPattern.Full.value,
        )

    @property
    def Distributed(self):
        """Options specific to distributed execution."""
        return self._distributed

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
        """set the IPU SyncPattern to one of:
        - Full
        - SinglePipeline
        - ReplicaAndLadder
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

    def randomSeed(self, random_seed):
        """Set the seed for the random number generator on the IPU.
        """
        assert isinstance(random_seed, int)
        torch.manual_seed(random_seed)
        self.createOrSet(random_seed=random_seed)
        return self

    def toDict(self):
        """ Merge all the options, except for the Jit ones, into a single
        dictionary to be serialised and passed to the cpp side."""
        assert not self.defaultAnchorMode(
        ), "An anchor mode must be picked before serialisation"
        out = {}
        out.update(self._popart.options)
        out = self.update(out)
        out = self._training.update(out)
        out = self._distributed.update(out)
        config_file = self._distributed.getGcdConfigFile()
        if self._distributed.numHosts > 1 or config_file:
            assert config_file, ("No IPUoF configuration file found for "
                                 "hostId %d" % self._distributed.hostId)
            os.environ["IPUOF_CONFIG_PATH"] = config_file
            logger.debug("'IPUOF_CONFIG_PATH' set to %s for hostId %d",
                         config_file, self._distributed.hostId)

        return out
