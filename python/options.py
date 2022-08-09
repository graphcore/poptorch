# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import copy
from typing import Optional, Union, Dict, Any, List, Set
import torch
from . import autocasting
from . import enums
from ._logging import logger
from . import _options_config
from . import _options_impl
from . import ops


class Attribute():
    _current_attrs = {}

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._saved = {}

    def __enter__(self):
        self._saved = copy.deepcopy(Attribute._current_attrs)
        for attr, dictionary in self._kwargs.items():
            for k, v in dictionary.items():
                torch.ops.poptorch.set_attribute(attr, k, v)
            if attr in Attribute._current_attrs:
                Attribute._current_attrs[attr].update(dictionary)
            else:
                Attribute._current_attrs[attr] = dictionary

    def __exit__(self, type, value, traceback):
        for attr, dictionary in self._kwargs.items():
            saved_dict = self._saved.get(attr, {})
            for k in dictionary.keys():
                if k not in saved_dict:
                    torch.ops.poptorch.clear_attribute(attr, k)
                else:
                    torch.ops.poptorch.set_attribute(attr, k, saved_dict[k])
        Attribute._current_attrs = self._saved


# Used by _options_config, defined here so that it is reported
# to the user as a "poptorch.options.ConfigFileError"
class ConfigFileError(Exception):
    pass


class _JitOptions(_options_impl.OptionsDict):
    """Options related to PyTorch's JIT compiler.

    Can be accessed via :py:attr:`poptorch.Options.Jit`:

    >>> opts = poptorch.Options()
    >>> opts.Jit.traceModel(True)
    """

    def __init__(self) -> None:
        super().__init__(trace_model=True)

    def traceModel(self, trace_model: bool) -> "poptorch.options._JitOptions":
        """
        Controls whether to use PyTorch's JIT tracing or the dispatcher
        to build the PopTorch graph.

        The support for the dispatcher is still experimental, therefore
        by default PopTorch will use torch.jit.trace().

        :param bool trace_model:
            * True: use `torch.jit.trace <https://pytorch.org/docs/1.10.0/generated/torch.jit.trace.html#torch.jit.trace>`_
            * False: use Torch's dispatcher to trace the graph.
       """
        self.set(trace_model=trace_model)
        return self


class _PrecisionOptions(_options_impl.OptionsDict):
    """ Options related to processing the PyTorch JIT graph prior to lowering to
    PopART

    Can be accessed via :py:attr:`poptorch.Options.Precision`:

    >>> opts = poptorch.Options()
    >>> opts.Precision.halfFloatCasting(
    ...   poptorch.HalfFloatCastingBehavior.HalfUpcastToFloat)
    """

    def __init__(self,
                 popart_options: "poptorch.options._PopartOptions") -> None:
        self._popart_options = popart_options
        super().__init__(
            autocast_enabled=True,
            autocast_policy=autocasting.default,
            autocast_policy_dict=autocasting.default._dict(),  # pylint: disable=protected-access
            half_float_casting=enums.HalfFloatCastingBehavior.Default)

    def autocastEnabled(self, autocast_enabled: bool
                        ) -> "poptorch.options._PrecisionOptions":
        """ Controls whether automatic casting functionality is turned on.

            :param autocast_enabled: if True, automatic casting is active.
                                          Default value is True.
        """

        if not isinstance(autocast_enabled, bool):
            raise ValueError(
                'autocastEnabled must be set to either True or False')

        self.set(autocast_enabled=autocast_enabled)
        return self

    def autocastPolicy(self, autocast_policy: "poptorch.autocasting.Policy"
                       ) -> "poptorch.options._PrecisionOptions":
        """ Set the automatic casting policy.

            :param policy: the policy object.
        """

        if not isinstance(autocast_policy, autocasting.Policy):
            raise ValueError('autocastPolicy must be set to an instance of'
                             'poptorch.autocasting.Policy')

        self.set(autocast_policy=autocast_policy)
        self.set(autocast_policy_dict=self.autocast_policy._dict())  # pylint: disable=protected-access
        return self

    def halfFloatCasting(
            self, half_float_casting: "poptorch.HalfFloatCastingBehavior"
    ) -> "poptorch.options._PrecisionOptions":
        """ Changes the casting behaviour for ops involving a float16 (half) and
            a float32

        The default option, ``Default``, is interpreted differently depending
        whether tracing is enabled or not.

        With tracing disabled, mixed precision casting always follows PyTorch's
        scheme, wherein all parameters are upcast to the type of the highest
        precision input. The exception to this is inplace ops, which cast
        to the precision of the output. This behaviour can also be specified
        explicitly using ``HalfUpcastToFloat``.

        With tracing enabled, ``Default`` will cause mixed precision ops to
        downcast their inputs to float16. This behaviour can also be obtained
        with the explicit setting ``FloatDowncastToHalf``. The alternative is
        to set the option as ``HalfUpcastToFloat``, which will give the PyTorch
        default behaviour outlined above.

        ``FloatDowncastToHalf`` is only valid with tracing enabled.

        :param half_float_casting:
            * ``FloatDowncastToHalf``:  Any op with operands (inputs) which are
              a mix of float32 and float16 (half) will cast all operands to
              half.
            * ``HalfUpcastToFloat``: Implicit casting will follow PyTorch's
              rules, promoting float16 (half) inputs to float32 if another input
              is float32.
            * ``Default``: Interpreted as ``FloatDowncastToHalf`` if tracing is
              enabled, or ``HalfUpcastToFloat`` if tracing is disabled.
        """

        if not isinstance(half_float_casting, enums.HalfFloatCastingBehavior):
            raise ValueError(
                "halfFloatCasting must be set to "
                "poptorch.HalfFloatCastingBehavior.FloatDowncastToHalf or "
                "poptorch.HalfFloatCastingBehavior.HalfUpcastToFloat or "
                "poptorch.Default")

        self.set(half_float_casting=half_float_casting)
        return self

    def runningStatisticsAlwaysFloat(self, value: bool
                                     ) -> "poptorch.options._PrecisionOptions":
        """Controls whether the running mean and variance tensors of batch
        normalisation layers should be float32 regardless of input type.

        A batch normalisation layer stores a running estimate of the means and
        variances of each channel during training, for use at inference in lieu
        of batch statistics. Storing the values as half (float16) can result in
        poor performance due to the low precision. Enabling this option yields
        more reliable estimates by forcing all running estimates of variances to
        be stored as float32, at the cost of extra memory use.

        :param value:
            * True: Always store running estimates of mean and variance as
              float32.
            * False: Store running estimates of mean and variance as the same
              type as the layer input.
        """

        if not isinstance(value, bool):
            raise ValueError(
                "runningStatisticsAlwaysFloat needs to be set to a bool")

        self.createOrSet(running_statistics_always_float=value)
        return self

    def enableFloatingPointExceptions(
            self, enabled: bool) -> "poptorch.options._PrecisionOptions":
        """Set whether floating point exceptions are enabled on the IPU.

        When enabled, an exception will be generated when the IPU encounters
        any one of the following:

        * Operation resulting in subtraction of infinities
        * Divisions by zero or by infinity
        * Multiplications between zero and infinity
        * Real operations producing complex results
        * Comparison where any one operand is Not-a-Number

       :param enabled:
           * True: raise ``RuntimeError`` on floating point exception
           * False: do not raise ``RuntimeError`` (default)
        """

        assert isinstance(enabled, bool), \
            "enableFloatingPointExceptions needs to be set to a bool"

        self._popart_options.set("enableFloatingPointChecks", enabled)
        return self

    def enableStochasticRounding(self, enabled: bool
                                 ) -> "poptorch.options._PrecisionOptions":
        """Set whether stochastic rounding is enabled on the IPU.

        Stochastic rounding rounds up or down a values to half (float16)
        randomly such that that the expected (mean) result of rounded value is
        equal to the unrounded value. It can improve training performance by
        simulating higher precision behaviour and increasing the speed or
        likelihood of model convergence. However, the model is non-deterministic
        and represents a departure from (deterministic) standard IEEE FP16
        behaviour.

        In the general case, we recommend enabling stochastic rounding for
        training where convergence is desirable, but not for inference where
        non-determinism may be undesirable.

        :param enabled:
            * True: Enable stochastic rounding on the IPU.
            * False: Disable stochastic rounding.
        """
        self._popart_options.set("enableStochasticRounding", enabled)
        return self

    def setPartialsType(self, dtype: torch.dtype
                        ) -> "poptorch.options._PrecisionOptions":
        """Set the data type of partial results for matrix multiplication and
        convolution operators.

        The matrix multiplication and convolution operators store intermediate
        results known as partials as part of the calculation. You can use this
        option to change the data type of the partials. Using ``torch.half``
        reduces on-chip memory use at the cost of precision.


        :param torch.dtype type:
            The type to store partials, which must be either ``torch.float`` or
            ``torch.half``
        """

        type_str = ''
        if dtype in [torch.float, torch.float32]:
            type_str = 'float'
        elif dtype in [torch.half, torch.float16]:
            type_str = 'half'
        else:
            raise ValueError("parameter to setPartialsType should be either" \
                             "torch.float or torch.half")

        self._popart_options.set("partialsTypeMatMuls", type_str)
        self._popart_options.set("convolutionOptions",
                                 {"partialsType": type_str})
        return self


class _TrainingOptions(_options_impl.OptionsDict):
    """Options specific to model training.

    .. note:: You must not set these options for inference models.

    Can be accessed via :py:attr:`poptorch.Options.Training`:

    >>> opts = poptorch.Options()
    >>> opts.Training.gradientAccumulation(4)
    """

    def __init__(self,
                 popart_options: "poptorch.options._PopartOptions") -> None:
        self._popart_options = popart_options
        super().__init__(gradient_accumulation=1,
                         accumulation_and_replication_reduction_type=enums.
                         ReductionType.Mean,
                         meanAccumulationAndReplicationReductionStrategy=enums.
                         MeanReductionStrategy.Post)

    def gradientAccumulation(self, gradient_accumulation: int
                             ) -> "poptorch.options._TrainingOptions":
        """Number of micro-batches to accumulate for the gradient calculation.

        Accumulate the gradient ``gradient_accumulation`` times before updating
        the model using the gradient. Other frameworks may refer to this setting
        as "pipeline depth".

        Accumulate the gradient ``gradient_accumulation`` times before updating
        the model using the gradient. Each micro-batch (a batch of size equal to
        the  ``batch_size`` argument passed to
        :py:class:`poptorch.DataLoader`) corresponds to one gradient
        accumulation. Therefore ``gradient_accumulation`` scales the global
        batch size (number of samples between optimiser     updates).

        .. note:: Increasing ``gradient_accumulation`` does not alter the
            (mini-) batch size used for batch normalisation.

        A large value for ``gradient_accumulation`` can improve training
        throughput by amortising optimiser update costs, most notably when using
        :py:class:`~poptorch.PipelinedExecution` or when training is distributed
        over a number of replicas. However, the consequential increase in the
        number of samples between optimiser updates can have an adverse impact
        on training.

        The reason why the efficiency gains are most notable when training with
        models with multiple IPUs which express pipelined model parallelism
        (via :py:class:`~poptorch.PipelinedExecution` or by default and
        annotating the model :py:class:`poptorch.BeginBlock` or
        :py:class:`poptorch.Block`) is because the pipeline has "ramp up" and
        "ramp down" steps around each optimiser update. Increasing the
        gradient accumulation factor in this instance reduces the proportion of
        time spent in the "ramp up" and "ramp down" phases, increasing overall
        throughput.

        When training involves multiple replicas, including the cases of sharded
        and phased execution, each optimiser step incurs a communication cost
        associated with the reduction of the gradients. By accumulating
        gradients, you can reduce the total number of updates required and thus
        reduce the total amount of communication.

        .. note::  Increasing the global batch size can have adverse effects on
           the sample efficiency of training so it is recommended to use a low
           or unity gradient accumulation count initially, and then try
           increasing to achieve higher throughput. You may also need to scale
           other hyper-parameters such as the optimiser learning rate
           accordingly.
        """

        self.set(gradient_accumulation=gradient_accumulation)

        return self

    def _check_reduction_arg(self, reduction_type, name):
        incorrect_instance = not isinstance(reduction_type,
                                            enums.ReductionType)
        no_red = reduction_type == enums.ReductionType.NoReduction
        if incorrect_instance or no_red:
            raise ValueError(name + " must be set to "
                             "poptorch.ReductionType.Mean or "
                             "poptorch.ReductionType.Sum")

    def accumulationAndReplicationReductionType(
            self, reduction_type: "poptorch.ReductionType"
    ) -> "poptorch.options._TrainingOptions":
        """Set the type of reduction applied to reductions in the graph.

        When using, a value for greater than one for
        :py:func:`~poptorch.options._TrainingOptions.gradientAccumulation` or
        for :py:func:`~poptorch.Options.replicationFactor`, PopTorch applies a
        reduction to the gradient outputs from each replica, and to the
        accumulated gradients. This reduction is independent of the model loss
        reduction (summing a mean-reduced loss and a sum-reduced loss in a
        PyTorch model is valid).

        This setting governs both the accumulation of the loss gradients in
        replicated graphs and of all of the gradients when using gradient
        accumulation.

        :param reduction_type:
            * Mean (default): Reduce gradients by calculating the mean of them.
            * Sum: Reduce gradients by calculating the sum of them.
        """
        self._check_reduction_arg(reduction_type,
                                  "accumulationAndReplicationReductionType")

        self.set(accumulation_and_replication_reduction_type=reduction_type)
        self._warnings_disabled.add(
            "accumulation_and_replication_reduction_type")
        return self

    def setMeanAccumulationAndReplicationReductionStrategy(
            self, mean_reduction_strategy: "poptorch.MeanReductionStrategy"
    ) -> "poptorch.options._TrainingOptions":
        """Specify when to divide by a mean reduction factor when
        ``accumulationAndReplicationReductionType`` is set to
        ``ReductionType.Mean``.

        The default reduction strategy depends on the optimizer used. The
        default strategy is `Running` when the `accum_type` of the optimizer is
        set to half-precision (float16) format. Otherwise the `Post` strategy
        is used as this strategy is typically more performant but the `Post`
        strategy is less numerically robust.

        :param mean_reduction_strategy:
            * Running: Keeps the reduction buffer as the current mean. This is
              preferred for numerical stability as the buffer value is never
              larger than the magnitude of the largest micro batch gradient.
            * Post: Divides by the accumulationFactor and replicatedGraphCount
              after all of the gradients have been reduced. In some cases this
              can be faster then using Running, however is prone to overflow.
            * PostAndLoss (deprecated): Divides by the replicatedGraphCount
              before the backwards pass, performs the gradient reduction
              across micro batches, and then divides by the accumulationFactor.
              This is to support legacy behaviour and is deprecated.
        """
        self.set(meanAccumulationAndReplicationReductionStrategy=
                 mean_reduction_strategy)
        return self

    def setAutomaticLossScaling(self, enabled: bool
                                ) -> "poptorch.options._TrainingOptions":
        """Set whether automatic loss scaling is enabled on the IPU.

        When using float16/half values for activations, gradients, and weights,
        the loss value needs to be scaled by a constant factor to avoid
        underflow/overflow. This adjustment is known as loss scaling. This
        setting automatically sets a global loss scaling factor during training.

        Note: This is an experimental feature and may not behave as expected.

        :param enabled:
            * True: Enable automatic loss scaling on the IPU.
            * False: Disable automatic loss scaling.
        """
        self._popart_options.set("automaticLossScalingSettings.enabled",
                                 enabled)
        return self

    def setConvolutionDithering(self, enabled: bool
                                ) -> "poptorch.options._TrainingOptions":
        """Enable convolution dithering.

        If true, then convolutions with different parameters will be laid out
        from different tiles in an effort to improve tile balance in models.

        Use ``MultiConv`` to apply this option to specific set of convolutions.

        :param enabled:
            Enables or disables convolution dithering for all convolutions.
        """

        self._popart_options.set("convolutionOptions",
                                 {"enableConvDithering": enabled})
        return self


class _PopartOptions:
    """Options specific to the PopART backend.

    Only for advanced users.

    Most options from `popart.SessionOptions` can be set using this class.

    .. note:: there is no mapping for the various PopART enums so integers need
    to be used instead.

    Can be accessed via :py:attr:`poptorch.Options._Popart`:

    >>> opts = poptorch.Options()
    >>> opts._Popart.set("autoRecomputation", 3) # RecomputationType::Pipeline
    >>> opts._Popart.set("syntheticDataMode",
    >>>                  int(popart.SyntheticDataMode.RandomNormal))
    """

    def __init__(self) -> None:
        self._is_frozen = False
        self.options = {}
        self.set("instrumentWithHardwareCycleCounter", False)
        self.set("rearrangeAnchorsOnHost", False)

    def __deepcopy__(self, memory):
        copied_options = _PopartOptions()
        memory[id(self)] = copied_options
        for key, val in self.__dict__.items():
            if key == '_is_frozen':
                val = False
            setattr(copied_options, key, copy.deepcopy(val, memory))
        return copied_options

    def checkIsFrozen(self, option=None):
        # Skip check during object initialization.
        if hasattr(self, '_is_frozen'):
            if option != '_is_frozen' and self._is_frozen:
                raise AttributeError("Can't modify frozen Options")

    def set(self, key: str, value: Union[int, float, str, List[str], Set[str]]
            ) -> "poptorch.options._PopartOptions":
        self.checkIsFrozen()

        self.options[key] = value
        return self

    def setEngineOptions(self, engine_options: Dict[str, str]
                         ) -> "poptorch.options._PopartOptions":
        self.set('engineOptions', engine_options)
        return self

    def setPatterns(self, patterns: Dict[str, bool],
                    level: int = 2) -> "poptorch.options._PopartOptions":
        """Override the default patterns of PopART's compiler.

        :param patterns: Dictionary of pattern names to
            enable / disable.
        :param level: Integer value corresponding to the
            ``popart::PatternsLevel`` to use to initialise the ``Patterns``.
        """
        assert isinstance(level, int)
        assert isinstance(patterns, dict)
        self.set("patterns_level", level)
        self.set("patterns", patterns)
        return self

    def __repr__(self):
        repr_body = ", ".join(f"{k}={v.__repr__()}"
                              for k, v in self.options.items())
        return f"{type(self).__name__}({repr_body})"


class _DistributedOptions(_options_impl.OptionsDict):
    """Options related to distributed execution.

    You should not use these when using PopRun/PopDist. Instead use
    ``popdist.poptorch.Options`` to set these values automatically.

    Can be accessed via :py:attr:`poptorch.Options.Distributed`:

    >>> opts = poptorch.Options()
    >>> opts.Distributed.configureProcessId(0, 2)
    """

    def __init__(self) -> None:
        self._gcd_mappings = {}
        super().__init__(num_distributed_processes=1,
                         distributed_process_id=0,
                         ipuof_configs={})
        self.setEnvVarNames("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK")

    def disable(self) -> "poptorch.options._DistributedOptions":
        """Ignore the current options / environment variables and disable
        distributed execution.
        """
        self.set(num_distributed_processes=1, distributed_process_id=0)
        return self

    def setEnvVarNames(self, var_num_processes: str, var_process_id: str
                       ) -> "poptorch.options._DistributedOptions":
        """Utility to read and set `processId` and `numProcesses` from
        environment variables.

        Useful if you use a third party library to manage the processes used for
        the distributed execution such as mpirun.

        For example: ``mpirun -np 4 myscript.py``

        By default the OpenMPI ``OMPI_COMM_WORLD_SIZE`` and
        ``OMPI_COMM_WORLD_RANK`` variables are used.
        """
        return self.configureProcessId(
            int(os.environ.get(var_process_id, "0")),
            int(os.environ.get(var_num_processes, "1")))

    def configureProcessId(self, process_id: int, num_processes: int
                           ) -> "poptorch.options._DistributedOptions":
        """Manually set the current process ID and the total number of processes.

        :param int process_id: The ID of this process.
        :param int num_processes: The total number of processes the execution is
            distributed over.
        """
        self.set(distributed_process_id=process_id)
        self.set(num_distributed_processes=num_processes)
        return self

    @property
    def processId(self) -> int:
        """Id of the current process."""
        return self.distributed_process_id

    @property
    def numProcesses(self) -> int:
        """Total number of processes the execution is distributed over."""
        return self.num_distributed_processes


class TensorLocationSettings(_options_impl.OptionsDict):
    """Define where a tensor is stored

    >>> opts = poptorch.Options()
    >>> opts.TensorLocations.setActivationLocation(
    ...     poptorch.TensorLocationSettings().useOnChipStorage(False))
    """

    def minElementsForOffChip(self, min_elements: int
                              ) -> "poptorch.TensorLocationSettings":
        """A minimum number of elements below which offloading
        won't be considered."""
        assert isinstance(min_elements, int)
        self.createOrSet(minElementsForOffChip=min_elements)
        return self

    def minElementsForReplicatedTensorSharding(
            self, min_elements: int) -> "poptorch.TensorLocationSettings":
        """Only enable replicated tensor sharding (RTS) for tensors with more
        than `min_elements` elements."""
        assert isinstance(min_elements, int)
        self.createOrSet(minElementsForReplicatedTensorSharding=min_elements)
        return self

    def useOnChipStorage(self, use: bool = True
                         ) -> "poptorch.TensorLocationSettings":
        """Permanent tensor storage

        :param bool use:
            True: use on chip memory.
            False: use off chip memory.
            None: keep it undefined.
        """
        if use is None:
            self.deleteIfExists("onChip")
        else:
            assert isinstance(use, bool)
            self.createOrSet(onChip=int(use))
        return self

    def useReplicatedTensorSharding(self, use: bool = True
                                    ) -> "poptorch.TensorLocationSettings":
        """Enable replicated tensor sharding

        (relevant for weights and optimiser states)
        """
        assert isinstance(use, bool)
        self.createOrSet(useReplicatedTensorSharding=int(use))
        return self

    def useIOTilesToLoad(self, use: bool = True
                         ) -> "poptorch.TensorLocationSettings":
        """Load tensor through IO tiles

        :param use: Use IO tiles if True,
                    use Compute tiles if False.
        """
        assert isinstance(use, bool)
        self.createOrSet(useIOTilesToLoad=int(use))
        return self

    def useIOTilesToStore(self, use: bool = True
                          ) -> "poptorch.TensorLocationSettings":
        """Use IO tiles to store tensors.

        (relevant for replicated tensor sharded tensors)

        :param use: Use IO tiles if True,
                    use Compute tiles if False.
        """
        assert isinstance(use, bool)
        self.createOrSet(useIOTilesToStore=int(use))
        return self


class _TensorLocationOptions(_options_impl.OptionsDict):
    """Options controlling where to store tensors.

    Can be accessed via :py:attr:`poptorch.Options.TensorLocations`:

    >>> opts = poptorch.Options()
    >>> opts.TensorLocations.setActivationLocation(
    ...     poptorch.TensorLocationSettings().useOnChipStorage(False))
    """

    def numIOTiles(self, num_tiles: int) -> "poptorch.TensorLocationSettings":
        """ Assigns the number of tiles on the IPU to be IO rather than compute.

        Allocating IO (input/output) tiles reduces the number of IPU tiles
        available for computation but allows you to reduce the latency of
        copying tensors from host to the IPUs using the function
        :py:func:`poptorch.set_overlap_for_input`, IPUs to host using the
        function
        :py:func:`poptorch.set_overlap_for_output` or to use off-chip memory
        with reduced by setting the option
        :py:meth:`~poptorch.TensorLocationSettings.useIOTilesToLoad`.
        As reducing the number of computation tiles may reduce peformance, you
        should not use any IO tiles until you have successfully run your model
        and used profiling to identify "streamCopy" entries which take up a
        significant proportion of execution time.
        """
        assert isinstance(num_tiles, int)

        err_msg = "numIOTiles must be an even number between 32 and 192."

        assert num_tiles >= 32, err_msg
        assert num_tiles <= 192, err_msg
        assert num_tiles % 2 == 0, err_msg

        self.createOrSet(numIOTiles=num_tiles)
        return self

    def setActivationLocation(self, location: "poptorch.TensorLocationSettings"
                              ) -> "poptorch.options._TensorLocationOptions":
        """
        :param location:
            Update tensor location settings for activations.
        """
        assert isinstance(location, TensorLocationSettings)
        self.createOrSet(location_activation=location.toDict())
        return self

    def setWeightLocation(self, location: "poptorch.TensorLocationSettings"
                          ) -> "poptorch.options._TensorLocationOptions":
        """
        :param location:
            Update tensor location settings for weights.
        """
        assert isinstance(location, TensorLocationSettings)
        self.createOrSet(location_weight=location.toDict())
        return self

    def setOptimizerLocation(self, location: "poptorch.TensorLocationSettings"
                             ) -> "poptorch.options._TensorLocationOptions":
        """
        :param location:
            Update tensor location settings for optimiser states.
        """
        assert isinstance(location, TensorLocationSettings)
        self.createOrSet(location_optimizer=location.toDict())
        return self

    def setAccumulatorLocation(self,
                               location: "poptorch.TensorLocationSettings"
                               ) -> "poptorch.options._TensorLocationOptions":
        """
        :param poptorch.TensorLocationSettings location:
            Update tensor location settings for accumulators.
        """
        assert isinstance(location, TensorLocationSettings)
        self.createOrSet(location_accumulator=location.toDict())
        return self


BlockId = str


class Stage:
    """
    The various execution strategies are made of `Stages`: a stage consists of
    one of more `Blocks` running on one IPU.

    .. seealso:: :py:class:`PipelinedExecution`, :py:class:`ShardedExecution`,
        :py:class:`ParallelPhasedExecution`, :py:class:`SerialPhasedExecution`.
    """

    def __init__(self, *block_ids: BlockId) -> None:
        assert all(isinstance(b, str) for b in block_ids), (
            "Block IDs are "
            f"supposed to be strings but got {block_ids}")
        self._blocks = block_ids
        self._stage_id = -1
        self._phase_id = -1
        self._ipu = None

    @property
    def blocks(self) -> List[BlockId]:
        """List of blocks this stage is made of."""
        return self._blocks

    def ipu(self, ipu: int) -> "poptorch.Stage":
        """Set the IPU on which this stage will run"""
        assert isinstance(ipu, int)
        self._ipu = ipu
        return self

    def _setStage(self, stage: int) -> "poptorch.Stage":
        if stage is not None:
            self._stage_id = stage
        return self


class _DefaultStageManager(_options_impl.IStageManager):
    def __init__(self, auto_stage: "poptorch.AutoStage") -> None:
        super().__init__()
        self._next_id = 1
        self._block_map = {}
        self._auto_stage = auto_stage

    def getStage(self, block_id: BlockId) -> "poptorch.Stage":
        if block_id not in self._block_map:
            stage = Stage(block_id)
            if self._auto_stage == enums.AutoStage.SameAsIpu:
                assert self._current_ipu is not None, (
                    f"poptorch.AutoStage.SameAsIpu was selected but no "
                    f"IPU was specified for block {block_id}")
                stage_id = self._current_ipu
            else:
                stage_id = self._next_id
                self._next_id += 1

            stage._setStage(stage_id)  # pylint: disable=protected-access
            self._block_map[block_id] = stage
        return self._block_map[block_id]


class _IExecutionStrategy:
    def __init__(self, stages_manager, block_map):
        self._block_map = block_map
        self._stages_manager = stages_manager

    def stage(self, block_id):
        """Return the :py:class:`poptorch.Stage` the given block is belongs to.

        :param str block_id: A block ID.
        """
        assert block_id in self._block_map, f"Unknown block {block_id}"
        return self._block_map[block_id]

    def onStartTracing(self):
        self._stages_manager.clearDebug()
        ops.Block._stages_manager = self._stages_manager  # pylint: disable=protected-access

    def onEndTracing(self):
        self._stages_manager.printDebug()
        ops.Block._stages_manager = None  # pylint: disable=protected-access

    def backendOptions(self):
        return {}


class Phase:
    """Represents an execution phase"""

    def __init__(self, *arg: Union[BlockId, "poptorch.Stage"]):
        """ Create a phase.

        :param arg: must either be one or more
            :py:class:`Stages<poptorch.Stage>`, or one or more
            blocks ``user_id``.

        If one or more strings are passed they will be interpreted as
        :py:class:`Block` IDs representing a single :py:class:`Stage`.

        Within a ``Phase``, the stages will be executed in parallel.

        >>> with poptorch.Block("A"):
        ...     layer()
        >>> with poptorch.Block("B"):
        ...     layer()
        >>> p = Phase(poptorch.Stage("A").ipu(0))
        >>> # 2 stages made of one block each
        >>> p = Phase(poptorch.Stage("A").ipu(0), poptorch.Stage("B").ipu(1))
        >>> p = Phase("A","B") # One Stage made of 2 blocks
        """
        if all([isinstance(elt, Stage) for elt in arg]):
            self.stages = arg
        else:
            assert all([isinstance(elt, str) for elt in arg
                        ]), ("All arguments must either "
                             "be block IDs (strings) or Stages: " +
                             str([type(elt) for elt in arg]))
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
    def __init__(self, *args):
        """Pipeline the execution of the graph partitions.
        These partitions can be:
        a :py:class:`Stage<poptorch.Stage>`, a :py:class:`Block<poptorch.Block>`
        or a :py:class:`BeginBlock<poptorch.BeginBlock>`.
        If none of these are passed, an :py:class:`poptorch.AutoStage` strategy
        can be passed instead to decide how the stage IDs are created.
        By default, `poptorch.AutoStage.SameAsIpu` is used: The stage ID
        will be set to the selected IPU number.
        This implies that each unique :py:class:`Block<poptorch.Block>` or
        :py:class:`BeginBlock<poptorch.BeginBlock>` in the graph must have
        their `ipu_id` explicitly set when using `AutoStage`.

        Example 1: Blocks `user_id` are known, IPUs are inferred.

        >>> with poptorch.Block("A"):
        ...     layer1()
        >>> with poptorch.Block("B"):
        ...     layer2()
        >>> with poptorch.Block("C"):
        ...     layer3()
        >>> with poptorch.Block("D"):
        ...     layer4()
        >>> opts = poptorch.Options()
        >>> # Create a 4 stages pipeline based on `user_id`, 4 IPUs will be used.
        >>> opts.setExecutionStrategy(poptorch.PipelinedExecution("A","B",
        ...                                                       "C","D"))

        Stages can also be set explicitly:

        >>> # Create a 2 stages pipeline with the blocks `user_id`, 2 IPUs will be used.
        >>> opts.setExecutionStrategy(poptorch.PipelinedExecution(
        ...    poptorch.Stage("A","B"),
        ...    poptorch.Stage("C","D")))

        Example 2: Blocks `ipu_id` are known, use default AutoStage.

        >>> poptorch.Block.useAutoId()
        >>> with poptorch.Block(ipu_id=0):
        ...     layer1()
        >>> with poptorch.Block(ipu_id=1):
        ...     layer2()
        >>> with poptorch.Block(ipu_id=2):
        ...     layer3()
        >>> with poptorch.Block(ipu_id=3):
        ...     layer4()
        >>> # Automatically create a 4-stage pipeline matching the block `ipu_id`.
        >>> opts.setExecutionStrategy(poptorch.PipelinedExecution())
        >>> # Note: poptorch.PipelinedExecution()
        >>> # is the default execution strategy when blocks are defined.

        Example 3:  Non-consecutive stages placed on the same IPU.

        >>> with poptorch.Block(ipu_id=0):
        ...     layer1()
        >>> with poptorch.Block(ipu_id=1):
        ...     layer2()
        >>> with poptorch.Block(ipu_id=0):
        ...     layer3()
        >>> # Automatically create a 3-stage pipeline forcing the stage
        >>> # IDs to be incremental.
        >>> opts.setExecutionStrategy(poptorch.PipelinedExecution(
        ...                           poptorch.AutoStage.AutoIncrement))

        :param args: Either a :py:class:`poptorch.AutoStage` strategy or an
            explicit list of stages or block IDs.
        :type args: poptorch.AutoStage, [str], [poptorch.Stage]

        """
        block_map = {}
        auto_stage = enums.AutoStage.SameAsIpu
        if len(args) == 1 and isinstance(args[0], enums.AutoStage):
            auto_stage = args[0]
        else:
            for stage_id, arg in enumerate(args):
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
                    assert block not in block_map, (
                        f"{block} associated "
                        f"with more than one stage")
                    logger.debug(
                        "block %s added to stage %d%s", block, stage_id,
                        " on IPU %d" %
                        stage._ipu if stage._ipu is not None else '')
                    block_map[block] = stage

        if block_map:

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
            stages_manager = _DefaultStageManager(auto_stage)
        super().__init__(stages_manager, block_map)

    def backendOptions(self):
        return {"execution_mode": 0}


class ShardedExecution(PipelinedExecution):
    """Will shard the execution of the passed Stages or if no stage is passed
    will consider each unique Block `ipu_id` encountered during tracing as a
    different stage.

    >>> with poptorch.Block(ipu_id=0):
    ...     layer()
    >>> with poptorch.Block(ipu_id=1):
    ...     layer()
    >>> with poptorch.Block(ipu_id=2):
    ...     layer()
    >>> opts = poptorch.Options()
    >>> # Automatically create 3 shards based on the block names
    >>> opts.setExecutionStrategy(poptorch.ShardedExecution())

    :param args: Either a :py:class:`poptorch.AutoStage` strategy or an
        explicit list of stages or block IDs.
    :type args: poptorch.AutoStage, [str], [poptorch.Stage]

    """

    def backendOptions(self):
        return {"execution_mode": 1}


class _IPhasedExecution(_IExecutionStrategy):
    """Common interface for Phased execution strategies"""

    def __init__(self, *phases: Union["poptorch.Phase", List["poptorch.Stage"],
                                      List[BlockId]]):
        """Execute the model's blocks in phases

        :param phases: Definition of phases must be either:

            - a list of :py:class:`poptorch.Phase`
            - a list of list of :py:class:`poptorch.Stage`
            - a list of list of :py:class:`poptorch.Block` IDs (Each list of
              blocks will be considered as a single :py:class:`poptorch.Stage` )
        :type phases: [:py:class:`poptorch.Phase`],
            [[:py:class:`poptorch.Stage`]], [[str]]

        """
        self._tensors_liveness = enums.Liveness.AlwaysLive
        self._separate_backward_phase = False
        self._phases = []
        block_map = {}
        for phase_id, args in enumerate(phases):
            if isinstance(args, Phase):
                phase = args
            else:
                if not isinstance(args, list):
                    args = [args]
                phase = Phase(*args)
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
            # TODO(T30127): Define what the default strategy should be.
            # stages_manager = _DefaultStageManager(enums.AutoStage.SameAsIpu)
            assert phases, (
                "There is currently no AutoStage for "
                "PhasedExecution, please explicitly specify the phases")

        super().__init__(stages_manager, block_map)

    def phase(self, phase: int) -> "poptorch.Phase":
        """Return the requested :py:class:`poptorch.Phase`

        :param phase: Index of the phase
        """
        assert isinstance(
            phase,
            int) and phase >= 0, "Phases are identified by positive integers"
        return self._phases[phase]

    def useSeparateBackwardPhase(self, use: bool = True):
        """Given a forward pass with 3 phases (0,1,2), by default the phases
        will run as follows: ::

            fwd:       bwd:
            phase 0 -> phase 4
            phase 1 -> phase 3
            phase 2 -> phase 2

        .. note:: The end of the forward pass and the beginning of the backward
            pass are part of the same phase.

        If ``useSeparateBackwardPhase(True)`` is used then no phase
        will be shared between the forward and backward passes: ::

            fwd:       bwd:
            phase 0 -> phase 6
            phase 1 -> phase 5
            phase 2 -> phase 4

        """
        assert isinstance(use, bool)
        self._separate_backward_phase = use
        return self

    def backendOptions(self) -> Dict[str, Union[int, bool]]:
        return {
            "execution_mode": 2,
            "separate_backward_phase": self._separate_backward_phase,
            "tensors_liveness": self._tensors_liveness.value
        }


class ParallelPhasedExecution(_IPhasedExecution):
    """Phases are executed in parallel alternating between two groups of IPUs.

    For example:

    - phase 0 runs on ipu 0 & 2
    - phase 1 runs on ipu 1 & 3
    - phase 2 runs on ipu 0 & 2

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
    >>> strategy = poptorch.ParallelPhasedExecution([
    ...     poptorch.Phase(poptorch.Stage("0"), poptorch.Stage("1")),
    ...     poptorch.Phase(poptorch.Stage("2"), poptorch.Stage("3")),
    ...     poptorch.Phase(poptorch.Stage("4"), poptorch.Stage("5"))])
    >>> strategy.phase(0).ipus(0,2)
    >>> strategy.phase(1).ipus(1,3)
    >>> strategy.phase(2).ipus(0,2)
    >>> opts.setExecutionStrategy(strategy)
    """

    def backendOptions(self) -> Dict[str, Union[int, bool]]:
        return {**super().backendOptions(), "serial_phases_execution": False}

    def sendTensorsOffChipAfterFwd(self, off_chip: bool = True
                                   ) -> "poptorch.ParallelPhasedExecution":
        assert isinstance(off_chip, bool)
        if off_chip:
            self._tensors_liveness = enums.Liveness.OffChipAfterFwd
        else:
            self._tensors_liveness = enums.Liveness.AlwaysLive
        return self


class SerialPhasedExecution(_IPhasedExecution):
    """All the phases run serially on a single group of IPUs.

    For example:

    - phase 0 runs on ipu 0 & 1
    - phase 1 runs on ipu 0 & 1
    - phase 2 runs on ipu 0 & 1

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

    def setTensorsLiveness(self, liveness: "poptorch.Liveness"
                           ) -> "poptorch.SerialPhasedExecution":
        """See :py:class:`poptorch.Liveness` for more information
        """
        assert isinstance(liveness, enums.Liveness)
        self._tensors_liveness = liveness
        return self

    def backendOptions(self) -> Dict[str, Union[int, bool]]:
        return {**super().backendOptions(), "serial_phases_execution": True}


# pylint: disable=too-many-public-methods
class Options(_options_impl.OptionsDict):
    """Set of all options controlling how a model is compiled and executed.

       Pass an instance of this class to the model wrapping functions
       :py:func:`poptorch.inferenceModel` and :py:func:`poptorch.trainingModel`
       to change how the model is compiled and executed. An instance includes
       general options set within this class such as
       :py:func:`poptorch.Options.deviceIterations` as
       well as properties referring to categories of options such as
       ``Training``.

        >>> opts = poptorch.Options()
        >>> opts.deviceIterations(10)
        >>> opts.Training.gradientAccumulation(4)

    """

    def __init__(self) -> None:
        self._jit = _JitOptions()
        self._popart = _PopartOptions()
        self._graphProcessing = _PrecisionOptions(self._popart)
        self._training = _TrainingOptions(self._popart)
        self._distributed = _DistributedOptions()
        self._tensor_locations = _TensorLocationOptions()
        self._execution_strategy = PipelinedExecution()
        # Don't pass it to super().__init__() -> we don't want it to be passed to the backend with the other
        # options. (It is passed to createGraph() instead).
        self._source_location_excludes = copy.copy(
            _options_impl.default_source_location_excludes)

        self.relaxOptimizerAttributesChecks(False)
        self.showCompilationProgressBar(True)
        self._module_namescope_enabled = True
        super().__init__(replication_factor=1,
                         broadcast_buffers=True,
                         device_iterations=1,
                         log_dir=".",
                         auto_round_num_ipus=False,
                         anchored_tensors={},
                         output_mode=enums.OutputMode.Default.value,
                         output_return_period=1,
                         connection_type=enums.ConnectionType.Always.value,
                         sync_pattern=enums.SyncPattern.Full.value,
                         available_memory_proportion={})
        path = os.environ.get("POPTORCH_CACHE_DIR", "")
        if path:
            logger.info("POPTORCH_CACHE_DIR is set: setting cache path to %s",
                        path)
            self.enableExecutableCaching(path)

    def sourceLocationExcludes(self,
                               excludes: List[str]) -> "poptorch.Options":
        """ When printing the IR all the frames containing one of the excluded
            strings will be ignored.

            This is helpful to get the IR to trace back to user code rather
            than some function inside a framework.

            :param excludes: Replace the current list of exclusions with this
                             one.
        """

        self._source_location_excludes = excludes
        return self

    def appendToLocationExcludes(self, *excludes: str) -> "poptorch.Options":
        """ When printing the IR all the frames containing one of the excluded
            strings will be ignored.

            This is helpful to get the IR to trace back to user code rather
            than some function inside a framework.

            :param excludes: Append these exclusions to the existing
                             list of exclusions.
        """
        self._source_location_excludes += excludes
        return self

    def showCompilationProgressBar(self,
                                   show: bool = True) -> "poptorch.Options":
        """Show / hide a progress bar while the model is being compiled.
        (The progress bar is shown by default)
        """
        self._show_compilation_progress_bar = show
        return self

    def loadFromFile(self, filepath: str) -> "poptorch.Options":
        """Load options from a config file where each line in the file
        corresponds to a single option being set. To set an option, simply
        specify how you would set the option within a Python script, but omit
        the ``options.`` prefix.

        For example, if you wanted to set ``options.deviceIterations(1)``,
        this would be set in the config file by adding a single line with
        contents ``deviceIterations(1)``.
        """
        _options_config.parseAndSetOptions(self, filepath)
        return self

    def relaxOptimizerAttributesChecks(self, relax: bool = True
                                       ) -> "poptorch.Options":
        """Controls whether unexpected attributes in
        :py:func:`~poptorch.PoplarExecutor.setOptimizer()` lead to warnings or
        debug messages.

        By default PopTorch will print warnings the first time it encounters
        unexpected attributes in
        :py:func:`~poptorch.PoplarExecutor.setOptimizer()`.

        :param relax:
            * True: Redirect warnings to the debug channel.
            * False: Print warnings about unexpected attributes (default
              behaviour).
        """
        # Doesn't need to be stored in the OptionsDict because it's only used
        # by the python side.
        self._relax_optimizer_checks = relax
        return self

    @property
    def TensorLocations(self) -> "poptorch.options._TensorLocationOptions":
        """Options related to tensor locations.

        .. seealso:: :py:class:`poptorch.options._TensorLocationOptions`"""
        return self._tensor_locations

    @property
    def Distributed(self) -> "poptorch.options._DistributedOptions":
        """Options specific to running on multiple IPU server (IPU-POD).

        You should not use these when using PopRun/PopDist. Instead use
        ``popdist.poptorch.Options`` to set these values automatically.

        .. seealso:: :py:class:`poptorch.options._DistributedOptions`"""
        return self._distributed

    @property
    def Jit(self) -> "poptorch.options._JitOptions":
        """Options specific to upstream PyTorch's JIT compiler.

        .. seealso:: :py:class:`poptorch.options._JitOptions`"""
        return self._jit

    @property
    def Precision(self) -> "poptorch.options._PrecisionOptions":
        """Options specific to the processing of the JIT graph prior to lowering
        to PopART.

        .. seealso:: :py:class:`poptorch.options._PrecisionOptions`"""
        return self._graphProcessing

    @property
    def Training(self) -> "poptorch.options._TrainingOptions":
        """Options specific to training.

        .. seealso:: :py:class:`poptorch.options._TrainingOptions`"""
        return self._training

    @property
    def _Popart(self) -> "poptorch.options._PopartOptions":
        """Options specific to the PopART backend.
        (Advanced users only)."""
        return self._popart

    def autoRoundNumIPUs(self, auto_round_num_ipus: bool = True
                         ) -> "poptorch.Options":
        """Whether or not to round up the number of IPUs used automatically: the
        number of IPUs requested must be a power of 2. By default, an error
        occurs if the model uses an unsupported number of IPUs
        to prevent you unintentionally overbooking IPUs.

        :param auto_round_num_ipus:
            * True: round up the number of IPUs to a power of 2.
            * False: error if the number of IPUs is not supported.

        """
        self.set(auto_round_num_ipus=auto_round_num_ipus)
        return self

    def deviceIterations(self, device_iterations: int) -> "poptorch.Options":
        """Number of iterations the device should run over the data before
        returning to the user (default: 1).

        This is equivalent to running the IPU in a loop over that the specified
        number of iterations, with a new batch of data each time. However,
        increasing ``deviceIterations`` is more efficient because the loop runs
        on the IPU directly.
        """
        self.set(device_iterations=device_iterations)
        return self

    def setExecutionStrategy(
            self, strategy: Union["poptorch.ParallelPhasedExecution",
                                  "poptorch.SerialPhasedExecution"]
    ) -> "poptorch.Options":
        """Set the execution strategy to use to partition the graph.

        :param strategy:
            Must be an instance of once of the execution strategy classes.

        .. seealso:: :py:class:`PipelinedExecution`,
            :py:class:`ShardedExecution`, :py:class:`ParallelPhasedExecution`,
            :py:class:`SerialPhasedExecution`.
        """
        assert isinstance(strategy, _IExecutionStrategy)
        self._execution_strategy = strategy
        return self

    def setAvailableMemoryProportion(
            self, available_memory_proportion: Dict[str, float]):
        """Sets the amount of temporary memory made available on a per-IPU basis.

        Use this setting to control the amount of temporary memory available to
        operations such as:

        * convolution
        * matrix multiplication
        * embedding lookups
        * indexing operations

        Parameter should be a dictionary of IPU IDs and float values between 0
        and 1. (for example, ``{"IPU0": 0.5}``)

        The floating point value has the same meaning and effect as documented
        in :py:func:`~poptorch.set_available_memory`.
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

    def replicationFactor(self, replication_factor: int) -> "poptorch.Options":
        """Number of times to replicate the model (default: 1).

        Replicating the model increases the data throughput of the model as
        PopTorch uses more IPUs. This leads to the number of IPUs used being
        scaled by ``replication_factor``, for example, if your model uses 1 IPU,
        a ``replication_factor`` of 2 will use 2 IPUs; if your model uses 4
        IPUs, a replication factor of 4 will use 16 IPUs in total.

        :param replication_factor:
            Number of replicas of the model to create.
        """
        self.set(replication_factor=replication_factor)
        return self

    def broadcastBuffers(self, broadcast_buffers: bool = True):
        """Broadcast buffers to all replicas.

        Only non-broadcast buffers are currently supported, which means each
        replica will hold a set of buffers not in sync with other replicas'
        buffers. To enable non-broadcast buffers, set this option to `False`.
        """
        self.set(broadcast_buffers=broadcast_buffers)
        return self

    def logDir(self, log_dir: str) -> "poptorch.Options":
        """Set the log directory

        :param log_dir:
            Directory where PopTorch saves log files (default: current
            directory)
        """
        self.set(log_dir=log_dir)
        return self

    def modelName(self, name: str) -> "poptorch.Options":
        """Set the model name

        :param name:
            Name of the model defaults to "inference" or "training" depending
            on the type of model created. Used when profiling to set the
            subdirectory of the report directory to output the profiling too.
        """
        self.createOrSet(model_name=name)
        return self

    def enableExecutableCaching(self, path: str) -> "poptorch.Options":
        """Load/save Poplar executables to the specified ``path``, using it as
        a cache,  to avoid recompiling identical graphs.

        :param path:
            File path for Poplar executable cache store; setting ``path`` to
            None`` disables executable caching.
        """
        if path is None:
            self._Popart.set("enableEngineCaching", False)
        else:
            self._Popart.set("cachePath", path)
            self._Popart.set("enableEngineCaching", True)
        return self

    def useIpuModel(self, use_model: bool) -> "poptorch.Options":
        """Whether to use the IPU Model or physical hardware (default)

        The IPU model simulates the behaviour of IPU hardware but does not offer
        all the functionality of an IPU. Please see the Poplar and PopLibs User
        Guide for further information.

        This setting takes precedence over the ``POPTORCH_IPU_MODEL``
        environment variable.

        :param use_model:
            * True: Use the IPU Model.
            * False: Use IPU hardware.
        """
        self.createOrSet(use_model=use_model)
        return self

    def connectionType(self, connection_type: "poptorch.ConnectionType"
                       ) -> "poptorch.Options":
        """When to connect to the IPU (if at all).

        :param connection_type:
            * ``Always``: Attach to the IPU from the start (default).
            * ``OnDemand``: Wait until the compilation is complete and the
              executable is ready to be run to attach to the IPU.
            * ``Never``: Never try to attach to an IPU: this is useful for
              offline compilation, but trying to run an executable will raise
              an exception.

        For example:

        >>> opts = poptorch.Options()
        >>> opts.connectionType(poptorch.ConnectionType.OnDemand)
        """
        assert isinstance(connection_type, enums.ConnectionType)
        self.set(connection_type=connection_type.value)
        return self

    def syncPattern(self, sync_pattern: "poptorch.SyncPattern"
                    ) -> "poptorch.Options":
        """Controls synchronisation in multi-IPU systems.

        This option can be used to allow subsets of IPUs to overlap their work.
        For example, one set of IPUs could be communicating with the host
        while other IPUs are processing data.

        This option is typically used together with replicated execution, in
        which case it takes effect on a per-replica basis. If replication is
        not used, it will apply to all IPUs.

        :param sync_pattern:
            * ``Full``: Require all IPUs to synchronise on every communication
              between IPUs or between IPUs and host. This is the default.
            * ``SinglePipeline``: Allow IPUs to synchronise with the host
              independently, without having to synchronise with each other.
              This permits any one IPU to perform host IO while other IPUs are
              processing data.
            * ``ReplicaAndLadder``: Allow an IPU group to communicate with the
              host without requiring synchronisation between groups. This
              permits multiple IPU groups to alternate between performing host
              IO and computation.
        """
        assert isinstance(sync_pattern, enums.SyncPattern)
        self.set(sync_pattern=sync_pattern.value)
        return self

    def useIpuId(self, ipu_id: int) -> "poptorch.Options":
        """ Use the IPU device specified by the ID (as provided by `gc-info`)

        A device ID may refer to a single or to a group of IPUs (a multi-IPU
        device). The number of IPUs associated with the ID must be equal to the
        number of IPUs used by your annotated model multiplied by the
        replication factor.

        For example if your model uses 1 IPU and the replication factor is 2
        you will need to provide a device ID with 2 IPU; if your model is
        pipelined across 4 IPUs and the replication factor is 4, you will need
        to provide a device ID which represents a multi-IPU device of 16 IPUs.

        You can use the the command-line tool `gc-info`: running `gc-info -a`,
        shows each device ID and a list of IPUs associated with the ID.

        :param int ipu_id: IPU device ID of a single-IPU or multi-IPU device
        """
        assert isinstance(ipu_id, int)
        self.createOrSet(ipu_id=ipu_id)
        return self

    def useOfflineIpuTarget(self, ipu_version: int = 2) -> "poptorch.Options":
        """Create an offline IPU target that can only be used for offline compilation.

        .. note:: the offline IPU target cannot be used if the IPU model is
            enabled.

        :param ipu_version: IPU version to target (1 for Mk1, 2 for Mk2).
            Default: 2.
        """
        self.connectionType(enums.ConnectionType.Never)
        self.createOrSet(ipu_version=ipu_version)
        return self

    def anchorTensor(self,
                     short_name: str,
                     long_name: str,
                     output_mode: Optional["poptorch.OutputMode"] = None,
                     output_return_period: Optional[int] = 1):
        """Anchor a tensor such that it may be retrieved after a model run.

        :param str short_name: User defined name to be used for retrieval
        :param str long_name: The PopART name of the tensor to be anchored
        :param poptorch.OutputMode output_mode: Specifies when data should
          be returned. Default to None, in which case the tensor will use
          the same output mode used for model outputs.
        :param int output_return_period: Return period if output mode is
          ``EveryN``. Defaults to 1.
        """

        if output_mode != enums.OutputMode.EveryN:
            output_return_period = 1

        value = [long_name, output_mode is None]
        value += [output_mode, output_return_period]
        self.anchored_tensors[short_name] = value

    def outputMode(self,
                   output_mode: "poptorch.OutputMode",
                   output_return_period: Optional[int] = None
                   ) -> "poptorch.Options":
        """ Specify which data to return from a model.

        :param poptorch.OutputMode output_mode:
            * ``All``: Return a result for each batch.
            * ``Sum``: Return the sum of all the batches.
            * ``Final``: Return the last batch.
            * ``EveryN``: Return every N batches: N is passed in
              as ``output_return_period``.
            * Default: `All` for inference, `Final` for training.

        For example:

        >>> opts = poptorch.Options()
        >>> opts.outputMode(poptorch.OutputMode.All)
        ... # or
        >>> opts.outputMode(poptorch.OutputMode.EveryN, 10)
        """
        assert isinstance(output_mode, enums.OutputMode)

        # Check the anchor return period makes sense.
        if output_mode == enums.OutputMode.EveryN:
            assert output_return_period and output_return_period > 0, (
                "EveryN"
                " anchor must have output_return_period set to valid"
                " positive integer")
        elif output_return_period:
            logger.info(
                "Anchor return period argument ignored with output_mode"
                " set to %s", output_mode)

        self.set(output_mode=output_mode.value,
                 output_return_period=output_return_period or 1)
        return self

    def defaultOutputMode(self) -> bool:
        """
        :return:
            * True: :py:func:`~poptorch.Options.outputMode` is currently set to
                default.
            * False: :py:func:`~poptorch.Options.outputMode` is not set to
                default.
        """
        return self.output_mode == enums.OutputMode.Default

    def randomSeed(self, random_seed: int) -> "poptorch.Options":
        """Set the seed for the random number generator on the IPU.

        :param random_seed:
            Random seed integer.
        """
        assert isinstance(random_seed, int)
        torch.manual_seed(random_seed)
        self.createOrSet(random_seed=random_seed)
        return self

    def enableStableNorm(self, enabled: bool) -> "poptorch.Options":
        """Set whether a stable version of norm operators is used.
        This stable version is slower, but more accurate than its
        unstable counterpart.

        :param enabled:
            * True: Use stable norm calculation.
            * False: Do not use stable norm calculation.
        """
        self._Popart.set("enableStableNorm", enabled)
        return self

    def enableSyntheticData(self, enabled: bool) -> "poptorch.Options":
        """Set whether host I/O is disabled and synthetic data
        is generated on the IPU instead. This can be used to benchmark
        models whilst simulating perfect I/O conditions.

        :param enabled:
            * True: Use data generated from a random normal distribution
              on the IPU. Host I/O is disabled.
            * False: Host I/O is enabled and real data is used.
        """
        # popart.SyntheticDataMode
        #   0 = Off
        #   1 = Zeros
        #   2 = RandomNormal
        mode = 2 if enabled else 0
        self._Popart.set("syntheticDataMode", mode)
        return self

    def logCycleCount(self, log_cycle_count: bool) -> "poptorch.Options":
        """Log the number of IPU cycles used in executing the main graph.

        The cycle count will be printed when this option is enabled by
        setting the environment variable ``POPTORCH_LOG_LEVEL=DEBUG``.
        This option requires IPU hardware to run.

        Note: This will have a small detrimental impact on performance.

        :param log_cycle_count:
            * True: Enable logging the IPU cycle count.
            * False: Do not enable IPU cycle count logging.
        """
        self._Popart.set("instrumentWithHardwareCycleCounter", log_cycle_count)
        return self

    def enableProfiling(self, profile_dir: Optional[str] = None
                        ) -> "poptorch.Options":
        """Enable profiling report generation.

        To generate debug information associated with the profiling
        data, please specify ``autoReport.directory``, and either
        ``autoReport.all`` or ``autoReport.outputDebugInfo`` in
        the ``POPLAR_ENGINE_OPTIONS`` environment variable. e.g.

        .. code-block:: bash

            POPLAR_ENGINE_OPTIONS={"autoReport.directory":"/profile/output",\\
            "autoReport.all":"true"}``

        or:

        .. code-block:: bash

            POPLAR_ENGINE_OPTIONS={"autoReport.directory":"/profile/output",\\
            "autoReport.outputDebugInfo":"true"}``

        Debug information and the rest of the profiling data will be stored in
        ``/profile/output directory``. Values specified in the environment
        variable take precedence over ``profile_dir`` when both are given.

        :param str profile_dir: path to directory where report will be created.
            Defaults to current directory.
        """
        env_engine_opts = os.getenv('POPLAR_ENGINE_OPTIONS', default='')
        env_override = ('debug.allowOutOfMemory' in env_engine_opts) or \
                       ('autoReport.directory' in env_engine_opts) or \
                       ('autoReport.all' in env_engine_opts)

        if env_override:
            logger.warning(
                'Profiling setting overriden by environment variable. '
                'Check content of POPLAR_ENGINE_OPTIONS.')

        opts = self._popart.options.get('engineOptions', {})
        opts['debug.allowOutOfMemory'] = 'true'
        opts['autoReport.directory'] = profile_dir or '.'
        opts['autoReport.all'] = 'true'
        self._popart.setEngineOptions(opts)
        return self

    def disableModuleNamescope(self) -> "poptorch.Options":
        """ Disable option adding name scope for each operator
        present in the module. This option is enabled by default.
        The operator name scope is be based on the names appearing
        in the named_modules function from torch.nn.Module.

        For example:

        >>> class Model(torch.nn.Module):
        >>>     def __init__(self, num_groups, num_channels):
        >>>         super().__init__()
        >>>         self.gn = torch.nn.GroupNorm(num_groups, num_channels)
        >>>     def forward(self, x):
        >>>         return self.gn2(x)

        With namescope enabled the name will be gn/GroupNormalization,
        with disabled it will be GroupNormalization.
        """
        self._module_namescope_enabled = False
        return self

    def toDict(self) -> Dict[str, Any]:
        """ Merge all the options, except for the JIT and Precision
        options, into a single dictionary to be serialised and passed to the C++
        backend.

        At this stage, any warnings are printed based on options set e.g. if
        a default option changes.

        :meta private:
        """
        assert not self.defaultOutputMode(
        ), "An output mode must be picked before serialisation"
        out = self._execution_strategy.backendOptions()
        out.update(self._popart.options)
        out = self.update(out)
        out = self._training.update(out)
        out = self._distributed.update(out)
        out = self._tensor_locations.update(out)

        if self._show_compilation_progress_bar:
            out["compilation_progress_bar_fn"] = _options_impl.ProgressBar()

        return out

    def clone(self) -> "poptorch.Options":
        """Create an unfrozen deep copy of the current options.
        """
        return copy.deepcopy(self)

    def __repr__(self):
        """Repr which recurses through the "properties" of the class to
        find the objects to print."""
        # Call __repr__ on v so that strings display with quotes.
        property_names = [
            p for p in dir(type(self))
            if isinstance(getattr(type(self), p), property)
        ]
        return (f"{type(self).__name__}(" +
                ", ".join(f"{k}={v.__repr__()}"
                          for k, v in self._values.items()) + ", " +
                ", ".join(f"{prop}={getattr(self, prop)}"
                          for prop in property_names) + ")")
