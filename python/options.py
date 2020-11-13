# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import glob
import json
import os
import torch
from . import enums
from .logging import logger
from . import _options_impl
from . import ops


class _JitOptions(_options_impl.OptionsDict):
    """Options related to Pytorch's JIT compiler.

    Can be accessed via `poptorch.Options`:

    >>> opts = poptorch.Options()
    >>> opts.Jit.traceModel(True)
    """

    def __init__(self):
        super().__init__(trace_model=True)

    def traceModel(self, trace_model):
        """
        If True: use torch.jit.trace
        If False: use torch.jit.script (Experimental)

        Trace model is enabled by default.
        """
        self.set(trace_model=trace_model)
        return self


class _TrainingOptions(_options_impl.OptionsDict):
    """Options specific to model training.

    Note: You must not set these options for inference models.

    Can be accessed via `poptorch.Options`:

    >>> opts = poptorch.Options()
    >>> opts.Training.gradientAccumulation(4)
    """

    def __init__(self):
        super().__init__(gradient_accumulation=1)

    def gradientAccumulation(self, gradient_accumulation):
        """Number of samples to accumulate for the gradient calculation.

        Accumulate the gradient N times before applying it. This is needed to
        train with models expressing pipelined model parallelism using the IPU
        annotation. This is due to weights being shared across pipeline batches
        so gradients will be updated and used by subsequent batches out of
        order.

        Might be called "pipeline depth" in some other frameworks."""
        self.set(gradient_accumulation=gradient_accumulation)
        return self


class _PopartOptions:
    """Options specific to the PopART backend.

    Only for advanced users.

    Any option from `popart.SessionOptions` can be set using this class.
    Note: there is no mapping for the various PopART enums so integers need
    to be used instead.

    Can be accessed via `poptorch.Options`:

    >>> opts = poptorch.Options()
    >>> opts.Popart.set("autoRecomputation", 3) # RecomputationType::Pipeline
    """

    def __init__(self):
        self.options = {}

    def set(self, key, value):
        self.options[key] = value
        return self

    def setPatterns(self, patterns, level=2):
        """Override the default patterns of Popart's compiler.

        :param dict(str,bool) patterns: Dictionary of pattern names to
            enable / disable.
        :param int level: Integer value corresponding to the
            ``popart::PaternsLevel`` to use to initialise the ``Patterns``.
        """
        assert isinstance(level, int)
        assert isinstance(patterns, dict)
        self.options["patterns_level"] = level
        self.options["patterns"] = patterns


class _DistributedOptions(_options_impl.OptionsDict):
    """Options related to distributed execution.

    Can be accessed via `poptorch.Options`:

    >>> opts = poptorch.Options()
    >>> opts.Distributed.configureProcessId(0, 2)
    """

    def __init__(self):
        super().__init__(num_distributed_processes=1,
                         distributed_process_id=0,
                         ipuof_configs={})
        self._gcd_mappings = {}
        self.setEnvVarNames("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK")

    def disable(self):
        """Ignore the current options / environment variables and disable
        distributed execution.
        """
        self.set(num_distributed_processes=1, distributed_process_id=0)
        return self

    def IPUoFConfigFiles(self, files):
        """ List of IPUoF configuration files to use for the different
        GCDs.

        Important: One and exactly one configuration file must be provided
        for each GCD.

        :param files: one or more glob compatible expressions

        By default: `~/.ipuof.conf.d/*.conf`

        The default value will work if you only own one partition.

        If you own several then you will need to narrow down the number of
        configuration files so that only the configuration files corresponding
        to the partition to use are selected.

        For example: `~/.ipuof.conf.d/partitionA_*.conf`
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

    def setEnvVarNames(self, var_num_processes, var_process_id):
        """Utility to read and set `processId` and `numProcesses` from
        environment variables.

        Useful if you use a third party library to manage the processes used for
        the distributed execution such as mpirun.

        For example: mpirun -np 4 myscript.py

        By default the OpenMPI "OMPI_COMM_WORLD_SIZE" and "OMPI_COMM_WORLD_RANK"
        variables are used.
        """
        return self.configureProcessId(
            int(os.environ.get(var_process_id, "0")),
            int(os.environ.get(var_num_processes, "1")))

    def configureProcessId(self, process_id, num_processes):
        """Manually set the current process ID and the total number of processess.

        :param int process_id: The ID of this process.
        :param int num_processes: The total number of processes the execution is
            distributed over.
        """
        self.set(distributed_process_id=process_id)
        self.set(num_distributed_processes=num_processes)
        return self

    def getGcdConfigFile(self):
        """Return all the GCD ids <-> file mappings.

        :meta private:
        """
        if not self._gcd_mappings:
            self.IPUoFConfigFiles("~/.ipuof.conf.d/*.conf")
        return self._gcd_mappings.get(self.processId)

    @property
    def processId(self):
        """Id of the current process."""
        return self.distributed_process_id

    @property
    def numProcesses(self):
        """Total number of processes the execution is distributed over."""
        return self.num_distributed_processes


class TensorLocationSettings(_options_impl.OptionsDict):
    def minElementsForOffChip(self, min_elements):
        """A minimum number of elements below which offloading
        won't be considered."""
        assert isinstance(min_elements, int)
        self.createOrSet(minElementsForOffChip=min_elements)
        return self

    def minElementsForReplicatedTensorSharding(self, min_elements):
        """Only enable Replicated Tensor Sharding (RTS) for tensors with more
        than `min_elements` elements."""
        assert isinstance(min_elements, int)
        self.createOrSet(minElementsForReplicatedTensorSharding=min_elements)
        return self

    def useOnChipStorage(self, use=True):
        """Permanent tensor storage

        :param bool use: True: use on chip memory,
                         False: use off chip memory.
                         None: keep it undefined.
        """
        if use is None:
            self.deleteIfExists("onChip")
        else:
            assert isinstance(use, bool)
            self.createOrSet(onChip=int(use))
        return self

    def useReplicatedTensorSharding(self, use=True):
        """Enable replicated tensor sharding

        (relevant for weights and optimizer states)
        """
        assert isinstance(use, bool)
        self.createOrSet(useReplicatedTensorSharding=int(use))
        return self

    def useIOTilesToLoad(self, use=True):
        """Load tensor through IO tiles

        :param bool use: Use IO tiles if True,
                         use Compute tiles if False.
        """
        assert isinstance(use, bool)
        self.createOrSet(useIOTilesToLoad=int(use))
        return self

    def useIOTilesToStore(self, use=True):
        """Use IO tiles to store tensors.

        (relevant for replicated tensor sharded tensors)

        :param bool use: Use IO tiles if True,
                         use Compute tiles if False.
        """
        assert isinstance(use, bool)
        self.createOrSet(useIOTilesToStore=int(use))
        return self


class _TensorLocationOptions(_options_impl.OptionsDict):
    """Options controlling where tensors are stored.
    """

    def setActivationLocation(self, location):
        assert isinstance(location, TensorLocationSettings)
        self.createOrSet(location_activation=location.toDict())
        return self

    def setWeightLocation(self, location):
        assert isinstance(location, TensorLocationSettings)
        self.createOrSet(location_weight=location.toDict())
        return self

    def setOptimizerLocation(self, location):
        assert isinstance(location, TensorLocationSettings)
        self.createOrSet(location_optimizer=location.toDict())
        return self

    def setAccumulatorLocation(self, location):
        assert isinstance(location, TensorLocationSettings)
        self.createOrSet(location_accumulator=location.toDict())
        return self


class Stage:
    """
    The various execution strategies are made of `Stages`: a stage consists of
    one of more `Blocks` running on one IPU.

    .. seealso:: :py:class:`PipelinedExecution`, :py:class:`ShardedExecution`,
        :py:class:`ParallelPhasedExecution`, :py:class:`SerialPhasedExecution`.
    """

    def __init__(self, *block_ids):
        self._blocks = block_ids
        self._stage_id = -1
        self._phase_id = -1
        self._ipu = None

    @property
    def blocks(self):
        """List of blocks this stage is made of."""
        return self._blocks

    def ipu(self, ipu):
        """Set the IPU on which this stage will run"""
        assert isinstance(ipu, int)
        self._ipu = ipu
        return self

    def _setStage(self, stage):
        if stage is not None:
            self._stage_id = stage
        return self


class _DefaultStageManager(_options_impl.IStageManager):
    def __init__(self):
        super().__init__()
        self._next_id = 0
        self._block_map = {}

    def getStage(self, block_id):
        if block_id not in self._block_map:
            stage = Stage(block_id)
            stage._setStage(self._default_stage or self._next_id)  # pylint: disable=protected-access
            self._next_id += 1
            self._block_map[block_id] = stage
        return self._block_map[block_id]


class _IExecutionStrategy:
    def __init__(self, stages_manager, block_map):
        self._block_map = block_map
        self._stages_manager = stages_manager

    def stage(self, block_id):
        assert block_id in self._block_map, f"Unknown block {block_id}"
        return self._block_map[block_id]

    def onStartTracing(self):
        ops.Block._stages_manager = self._stages_manager  # pylint: disable=protected-access

    def onEndTracing(self):
        ops.Block._stages_manager = None  # pylint: disable=protected-access

    def backendOptions(self):
        return {}


class Phase:
    def __init__(self, arg):
        """ arg must either be one or more Stages, or one or more block_ids.

        Within a Phase, the stages are executed in parallel.

        >>> with poptorch.Block("A"):
        ...     layer()
        >>> with poptorch.Block("B"):
        ...     layer()
        >>> p = Phase(poptorch.Stage("A").ipu(0))
        >>> # 2 stages made of one block each
        >>> p = Phase(poptorch.Stage("A").ipu(0), poptorch.Stage("B").ipu(1))
        >>> p = Phase("A","B") # One Stage made of 2 blocks
        """
        if isinstance(arg, (Stage, str)):
            arg = [arg]

        if all([isinstance(elt, Stage) for elt in arg]):
            self.stages = arg
        else:
            assert all([isinstance(elt, str)
                        for elt in arg]), ("All arguments"
                                           "must either be strings or Stages")
            self.stages = [Stage(*arg)]

    def stage(self, idx):
        return self.stages[idx]

    def ipus(self, *ipus):
        """Assign one IPU for each stage contained in this Phase.

        The number of IPUs passed must match the number of stages in the Phase.
        """
        assert len(ipus) == len(self.stages), (
            f"Phase contains "
            f"{len(self.stages)} stages but you provided {len(ipus)} ipus")
        for stage, ipu in zip(self.stages, ipus):
            stage.ipu(ipu)


class PipelinedExecution(_IExecutionStrategy):
    """Will pipeline the execution of the passed Stages or if no stage is passed
    will consider each unique Block name encountered during tracing as a
    different stage.
    >>> with poptorch.Block("A"):
    ...     layer()
    >>> with poptorch.Block("B"):
    ...     layer()
    >>> with poptorch.Block("C"):
    ...     layer()
    >>> opts = poptorch.Options()
    >>> # Create a 3 stages pipeline
    >>> opts.setExecutionStrategy(poptorch.PipelinedExecution("A","B","C"))
    >>> # Create a 2 stages pipeline
    >>> opts.setExecutionStrategy(poptorch.PipelinedExecution(
    ...    poptorch.Stage("A","B"),
    ...    "C"))
    >>> # Automatically create a 3 stages pipeline based on the block names
    >>> opts.setExecutionStrategy(poptorch.PipelinedExecution())
    """

    def __init__(self, *stages):
        block_map = {}
        for stage_id, arg in enumerate(stages):
            # arg must either be a Stage, a block_id or a list of block_ids
            if isinstance(arg, Stage):
                stage = arg
            elif isinstance(arg, str):
                stage = Stage(arg)
            else:
                assert all([isinstance(elt, str) for elt in arg])
                stage = Stage(*arg)
            stage._setStage(stage_id)  # pylint: disable=protected-access
            for block in stage.blocks:
                assert block not in block_map, (f"{block} associated "
                                                f"with more than one stage")
                logger.debug(
                    "block %s added to stage %d%s", block, stage_id,
                    " on IPU %d" %
                    stage._ipu if stage._ipu is not None else '')
                block_map[block] = stage
        if stages:

            class PipelineStageManager(_options_impl.IStageManager):
                def __init__(self, block_map):
                    super().__init__()
                    self._block_map = block_map

                def getStage(self, block_id):
                    assert block_id in self._block_map, (
                        f"Unknown Block "
                        f"'{block_id}' list of expected Blocks: "
                        f"{list(self._block_map.keys())}")
                    return self._block_map[block_id]

            stages_manager = PipelineStageManager(block_map)
        else:
            stages_manager = _DefaultStageManager()
        super().__init__(stages_manager, block_map)

    def backendOptions(self):
        return {"execution_mode": 0}


class ShardedExecution(PipelinedExecution):
    """Will shard the execution of the passed Stages or if no stage is passed
    will consider each unique Block name encountered during tracing as a
    different stage.
    >>> with poptorch.Block("A"):
    ...     layer()
    >>> with poptorch.Block("B"):
    ...     layer()
    >>> with poptorch.Block("C"):
    ...     layer()
    >>> opts = poptorch.Options()
    >>> # Automatically create 3 shards based on the block names
    >>> opts.setExecutionStrategy(poptorch.ShardedExecution())
    """

    def backendOptions(self):
        return {"execution_mode": 1}


class _IPhasedExecution(_IExecutionStrategy):
    """Common interface for Phased execution strategies"""

    def __init__(self, *phases):
        self._tensors_liveness = enums.Liveness.AlwaysLive
        self._separate_backward_phase = False
        self._phases = []
        block_map = {}
        for phase_id, stages in enumerate(phases):
            phase = Phase(stages)
            self._phases.append(phase)
            for _, stage in enumerate(phase.stages):
                stage._phase_id = phase_id
                for block in stage.blocks:
                    assert block not in block_map, (f"{block} associated "
                                                    "with more than one stage")
                    logger.debug(
                        "block %s added to phase %d%s", block, phase_id,
                        " on IPU %d" %
                        stage._ipu if stage._ipu is not None else '')
                    block_map[block] = stage
        if phases:

            class PhaseManager(_options_impl.IStageManager):
                def __init__(self, block_map):
                    super().__init__()
                    self._block_map = block_map

                def getStage(self, block_id):
                    assert block_id in self._block_map, (
                        f"Unknown Block "
                        f"'{block_id}' list of expected Blocks: "
                        f"{list(self._block_map.keys())}")
                    return self._block_map[block_id]

            stages_manager = PhaseManager(block_map)
        else:
            stages_manager = _DefaultStageManager()

        super().__init__(stages_manager, block_map)

    def phase(self, phase):
        assert isinstance(
            phase,
            int) and phase >= 0, "Phases are identified by positive integers"
        return self._phases[phase]

    def useSeparateBackwardPhase(self, use=True):
        """Given a forward pass with 3 phases (0,1,2), by default the phases
        will run as follow:

        fwd:       bwd:
        phase 0 -> phase 4
        phase 1 -> phase 3
        phase 2 -> phase 2

        Note: The end of the forward pass and the beginning of the backward
        pass are part of the same phase.

        If ``useSeparateBackwardPhase(True)`` is used then no phase
        will be shared between the forward and backward passes:

        fwd:       bwd:
        phase 0 -> phase 6
        phase 1 -> phase 5
        phase 2 -> phase 4
        """
        assert isinstance(use, bool)
        self._separate_backward_phase = use
        return self

    def backendOptions(self):
        return {
            "execution_mode": 2,
            "separate_backward_phase": self._separate_backward_phase,
            "tensors_liveness": self._tensors_liveness.value
        }


class ParallelPhasedExecution(_IPhasedExecution):
    """Phases are executed in parallel alternating between two groups of IPUs.

    For example:

    phase 0 runs on ipu 0 & 2
    phase 1 runs on ipu 1 & 3
    phase 2 runs on ipu 0 & 2

    >>> poptorch.Block.useAutoId()
    >>> with poptorch.Block(): # user_id = "0"
    ...     layer()
    >>> with poptorch.Block(): # user_id = "1"
    ...     layer()
    >>> with poptorch.Block(): # user_id = "2"
    ...     layer()
    >>> with poptorch.Block(): # user_id = "3"
    ...     layer()
    >>> with poptorch.Block(): # user_id = "4"
    ...     layer()
    >>> with poptorch.Block(): # user_id = "5"
    ...     layer()
    >>> opts = poptorch.Options()
    >>> strategy = poptorch.ParalellPhasedExecution([
    ...     poptorch.Phase(poptorch.Stage("0"), poptorch.Stage("1")),
    ...     poptorch.Phase(poptorch.Stage("2"), poptorch.Stage("3")),
    ...     poptorch.Phase(poptorch.Stage("4"), poptorch.Stage("5"))])
    >>> strategy.phase(0).ipus(0,2)
    >>> strategy.phase(1).ipus(1,3)
    >>> strategy.phase(2).ipus(0,2)
    >>> opts.setExecutionStrategy(strategy)
    """

    def backendOptions(self):
        return {**super().backendOptions(), "serial_phases_execution": False}

    def sendTensorsOffChipAfterFwd(self, off_chip=True):
        assert isinstance(off_chip, bool)
        if off_chip:
            self._tensors_liveness = enums.Liveness.OffChipAfterFwd
        else:
            self._tensors_liveness = enums.Liveness.AlwaysLive
        return self


class SerialPhasedExecution(_IPhasedExecution):
    """All the phases run serially on a single group of IPUs.

    For example:

    phase 0 runs on ipu 0 & 1
    phase 1 runs on ipu 0 & 1
    phase 2 runs on ipu 0 & 1

    >>> with poptorch.Block("A"):
    ...     layer()
    >>> with poptorch.Block("A2"):
    ...     layer()
    >>> with poptorch.Block("B"):
    ...     layer()
    >>> with poptorch.Block("B2"):
    ...     layer()
    >>> with poptorch.Block("C"):
    ...     layer()
    >>> with poptorch.Block("C2"):
    ...     layer()
    >>> opts = poptorch.Options()
    >>> strategy = poptorch.SerialPhasedExecution([
    ...     poptorch.Phase(poptorch.Stage("A"), poptorch.Stage("A2")),
    ...     poptorch.Phase(poptorch.Stage("B"), poptorch.Stage("B2")),
    ...     poptorch.Phase(poptorch.Stage("C"), poptorch.Stage("C2"))])
    >>> strategy.phase(0).ipus(0,1)
    >>> strategy.phase(1).ipus(0,1)
    >>> strategy.phase(2).ipus(0,1)
    >>> opts.setExecutionStrategy(strategy)
    """

    def setTensorsLiveness(self, liveness):
        """See ``Liveness`` for more information

        .. seealso:: :py:class:`poptorch.Liveness`
        """
        assert isinstance(liveness, enums.Liveness)
        self._tensors_liveness = liveness
        return self

    def backendOptions(self):
        return {**super().backendOptions(), "serial_phases_execution": True}


class Options(_options_impl.OptionsDict):
    """Options controlling how a model is run on the IPU.
    """

    def __init__(self):
        self._jit = _JitOptions()
        self._training = _TrainingOptions()
        self._popart = _PopartOptions()
        self._distributed = _DistributedOptions()
        self._tensor_locations = _TensorLocationOptions()
        self._execution_strategy = PipelinedExecution()

        super().__init__(replication_factor=1,
                         device_iterations=1,
                         log_dir=".",
                         anchor_mode=enums.AnchorMode.Default.value,
                         anchor_return_period=1,
                         use_model=False,
                         connection_type=enums.ConnectionType.Always.value,
                         sync_pattern=enums.SyncPattern.Full.value,
                         available_memory_proportion={})

    @property
    def TensorLocations(self):
        """Options related to tensor locations.

        .. seealso:: :py:class:`poptorch.options._TensorLocationOptions`"""
        return self._tensor_locations

    @property
    def Distributed(self):
        """Options specific to distributed execution.

        .. seealso:: :py:class:`poptorch.options._DistributedOptions`"""
        return self._distributed

    @property
    def Jit(self):
        """Options specific to upstream PyTorch's JIT compiler.

        .. seealso:: :py:class:`poptorch.options._JitOptions`"""
        return self._jit

    @property
    def Training(self):
        """Options specific to training.

        .. seealso:: :py:class:`poptorch.options._TrainingOptions`"""
        return self._training

    @property
    def Popart(self):
        """Options specific to the PopART backend.
        (Advanced users only).

        .. seealso:: :py:class:`poptorch.options._PopartOptions`"""
        return self._popart

    def deviceIterations(self, device_iterations):
        """Number of iterations the device should run over the data before
        returning to the user. (Default: 1)

        Essentially, it is the equivalent of launching the IPU in a loop over
        that number of batches. This is efficient because that loop runs
        on the IPU directly.
        """
        self.set(device_iterations=device_iterations)
        return self

    def setExecutionStrategy(self, strategy):
        """Set the execution strategy to use to partition the graph

        .. seealso:: :py:class:`PipelinedExecution`,
            :py:class:`ShardedExecution`, :py:class:`ParallelPhasedExecution`,
            :py:class:`SerialPhasedExecution`.
        """
        assert isinstance(strategy, _IExecutionStrategy)
        self._execution_strategy = strategy
        return self

    def setAvailableMemoryProportion(self, available_memory_proportion):
        """Memory is set on a per IPU basis, this should be a dictionary
        of IPU ids and float values between 0 and 1.

        For example: {"IPU0": 0.5}
        """
        actual_memory = {}

        for key, mem in available_memory_proportion.items():
            assert key.startswith("IPU"), (
                "Available memory proportions are expected"
                " to be in a dictionary of {\"IPU0\": 0.5}"
                " where the 0 in IPU is the index of the"
                " IPU. Invalid key: %s" % key)

            ipu_id = int(key[3:])
            actual_memory[ipu_id] = mem

        self.createOrSet(available_memory_proportion=actual_memory)
        return self

    def replicationFactor(self, replication_factor):
        """Number of model replications (Default: 1).

        For example if your model uses 1 IPU, a
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

        Default: False (Real Hardware).

        This setting takes precedence over the `POPTORCH_IPU_MODEL` environment
        variable.
        """
        self.set(use_model=use_model)
        return self

    def connectionType(self, connection_type):
        """set the IPU connection type to one of:

        :param poptorch.ConnectionType connection_type:
            * Always: Attach to the IPU from the start (Default).
            * OnDemand: Wait until the compilation is complete and the
              executable is ready to be run to attach to the IPU.
            * Never: Never try to attach to an IPU. (Useful for offline
              compilation, but trying to run an executable will raise
              an exception).
        """
        assert isinstance(connection_type, enums.ConnectionType)
        self.set(connection_type=connection_type.value)
        return self

    def syncPattern(self, sync_pattern):
        """Set the IPU SyncPattern.

        :param poptorch.SyncPattern sync_pattern:
            * Full
            * SinglePipeline
            * ReplicaAndLadder
        """
        assert isinstance(sync_pattern, enums.SyncPattern)
        self.set(sync_pattern=sync_pattern.value)
        return self

    def useIpuId(self, ipu_id):
        """ Use the specified IPU id as provided by `gc-info`.

        The number of IPUs associated with the id must be equal to the number
        of IPUs used by your grpah multiplied by the replication factor.

        For example if your model uses 1 IPU and the replication factor is 2
        you will need to provide an id with 2 IPUs.

        If your model is pipelined across 4 IPUs, the replication factor is 4,
        you will need to provide an id containing 16 IPUs total.

        :param int ipu_id: IPU id as provided by `gc-info`.
        """
        assert isinstance(ipu_id, int)
        self.createOrSet(ipu_id=ipu_id)
        return self

    def useOfflineIpuTarget(self, ipu_version=1):
        """Create an offline IPU target that can only be used for offline compilation.

        Note: the offline IPU target cannot be used if the IPU model is enabled.

        :param int ipu_version: IPU version to target (1 for mk1, 2 for mk2).
            Default: 1.
        """
        self.connectionType(enums.ConnectionType.Never)
        self.createOrSet(ipu_version=ipu_version)
        return self

    def anchorMode(self, anchor_mode, anchor_return_period=None):
        """ How much data to return from a model

        :param poptorch.AnchorMode anchor_mode:
            * All: Return a result for each batch.
            * Sum: Return the sum of all the batches
            * Final: Return the last batch.
            * EveryN: Return every N batches. N is passed in
              as ``anchor_return_period``.
            * Default: `All` for inference, `Final` for training.
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
        dictionary to be serialised and passed to the C++ backend.

        :meta private:
        """
        assert not self.defaultAnchorMode(
        ), "An anchor mode must be picked before serialisation"
        out = self._execution_strategy.backendOptions()
        out.update(self._popart.options)
        out = self.update(out)
        out = self._training.update(out)
        out = self._distributed.update(out)
        out = self._tensor_locations.update(out)
        config_file = self._distributed.getGcdConfigFile()
        if self._distributed.numProcesses > 1 or config_file:
            assert config_file, ("No IPUoF configuration file found for "
                                 "processId %d" % self._distributed.processId)
            os.environ["IPUOF_CONFIG_PATH"] = config_file
            logger.debug("'IPUOF_CONFIG_PATH' set to %s for processId %d",
                         config_file, self._distributed.processId)

        return out
