# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import enum
import numbers
import torch

# Do not import any poptorch.* here: it will break the poptorch module
from ._logging import logger
from . import optim, enums


class OptimizerAttrTracker:
    def __init__(self, opts):
        if opts._relax_optimizer_checks:
            self.log = logger.debug
        else:
            self.log = logger.warning
        self.group_attributes = []
        self.optim_attributes = []
        self.record_attributes = True
        self.printed_msgs = []
        self.type = "Unknown"

    def setType(self, optimizer_type):
        assert isinstance(optimizer_type, _OptimizerType)
        self.type = optimizer_type.name

    def enableChecks(self):
        self.record_attributes = False

    def checkDefaultAttributes(self, provided):
        self._check(self.group_attributes, provided, "default group variable")

    def checkGroupAttributes(self, provided, group):
        self._check(self.group_attributes, provided,
                    f"group {group} attribute")

    def checkOptimizerAttributes(self, provided):
        self._check(self.optim_attributes, provided, "optimizer attribute")

    def _check(self, expected, provided, attr_type):
        extra = [attr for attr in provided if attr not in expected]
        if self.record_attributes:
            expected += extra
        elif extra:
            msg = f"Ignoring unexpected {attr_type} in {self.type} optimizer:"
            msg += f" {extra}"
            if msg not in self.printed_msgs:
                self.log(msg)
                self.printed_msgs.append(msg)


# pylint: disable=too-many-statements
def convertOptimizerToDict(optimizer, attr_tracker, options, is_compiled):
    optimizer_type = _toPoptorchOptimizer(optimizer)
    attr_tracker.setType(optimizer_type)

    assert optimizer_type is not None, ("Unsupported optimizer type. "
                                        "Types supported %s") % str(
                                            list(_OptimizerType))
    opt_class = _toPoptorchClass(optimizer_type)

    num_groups = len(optimizer.param_groups)
    variable_attrs = getattr(optimizer, "variable_attrs", None)

    def assertNesterovDisabled(params):
        if params["nesterov"]:
            raise ValueError("Nesterov momentum is currently not supported.")
        return {}

    def assertAmsgradDisabled(params):
        if params["amsgrad"]:
            raise ValueError("Only non-amsgrad "
                             "Adam/AdamW optimizers are supported.")
        return {}

    def isFloat16(type, name):
        if type not in [torch.float16, torch.float32]:
            raise ValueError(f"{name} must be set to either torch.float16"
                             " or torch.float32 not {type}")
        return type == torch.float16

    def assertRMSProp(value, name):
        if optimizer_type not in (_OptimizerType.RMSPROP,
                                  _OptimizerType.RMSPROP_CENTERED):
            raise ValueError(
                f"{name} is only available with RMSProp optimizers.")
        return value

    def ignore(_params):
        return {}

    def isAlwaysConst(_value):
        return True

    def isNeverConst(_value):
        return False

    def isNotNaN(value, name):
        if value == float("nan"):
            raise ValueError(f"{name} must not be NaN")
        return value

    # Separate attributes which can be set per group (And therefore are stored
    # in `defaults` and `param_groups`) and the ones which are global and just
    # stored as attributes of the optimizer.

    # Register all the attribute readers
    attr_readers = {
        "nesterov": assertNesterovDisabled,
        "amsgrad": assertAmsgradDisabled,
        "bias_correction": ignore,
        "centered": ignore,
        "use_combined_accum": ignore
    }
    # Optimizer attributes: global, cannot change over time.
    #     source: opt.name
    #     format: {name: value}
    _AttrReader(attr_readers, "accum_type", _OptimizerGetter(torch.float32),
                isFloat16)
    _AttrReader(attr_readers, "velocity_accum_type",
                _OptimizerGetter(torch.float32), isFloat16)
    _AttrReader(attr_readers, "first_order_momentum_accum_type",
                _OptimizerGetter(torch.float32), isFloat16)
    _AttrReader(attr_readers, "second_order_momentum_accum_type",
                _OptimizerGetter(torch.float32), isFloat16)
    _AttrReader(attr_readers, "use_tf_variant", _OptimizerGetter(False),
                assertRMSProp)
    _AttrReader(attr_readers, "max_grad_norm", _OptimizerGetter(float("Inf")),
                isNotNaN)
    # Optimizer variables: global, can change over time.
    #     source: opt.name
    #     format: {name: (value, is_const)}

    # Set MeanReductionStrategy based on accum_type
    #     float32: Post (default)
    #     float16: Running
    if hasattr(optimizer,
               "accum_type") and optimizer.accum_type == torch.float16:
        # Only Post MeanReductionStrategy is supported for combined_accum variant
        if not hasattr(
                optimizer,
                "use_combined_accum") or not optimizer.use_combined_accum:
            if not is_compiled:
                # If the executable hasn't been compiled yet then it's ok to change
                # the reduction strategy.
                options._unfreeze()  # pylint: disable=protected-access
                options.Training.setMeanAccumulationAndReplicationReductionStrategy(  # pylint: disable=line-too-long
                    enums.MeanReductionStrategy.Running)
                options._freeze()  # pylint: disable=protected-access
            elif options.Training.meanAccumulationAndReplicationReductionStrategy != enums.MeanReductionStrategy.Running:  # pylint: disable=line-too-long
                raise ValueError(
                    "Invalid optimizer: the new optimizer would "
                    "require changing options.Training."
                    "meanAccumulationAndReplicationReductionStrategy to "
                    "poptorch.MeanReductionStrategy.Running but the "
                    "executable is already compiled.")

    # pylint: disable=protected-access
    auto_loss_scaling = options._Popart.options.get(
        "automaticLossScalingSettings.enabled", False)
    if variable_attrs and auto_loss_scaling:
        # Automatic loss scaling requires loss scaling to be variable
        variable_attrs.markAsVariable("loss_scaling")
    _AttrReader(
        attr_readers, "loss_scaling", _OptimizerGetter(1.0),
        _ValueConstPairFormatter(
            variable_attrs, lambda v: v == 1.0 and not auto_loss_scaling))
    # Group variables: per group, can change over time.
    #     source: opt.param_groups[i][name] / opt.defaults[name]
    #     format: {name: (value, is_const)}
    _AttrReader(attr_readers,
                "lr",
                _GroupGetter(),
                _ValueConstPairFormatter(variable_attrs, isNeverConst),
                new_name="learningRate")
    weight_decay_const_value = 0.0
    # In PyTorch AdamW has a different default value from Adam
    if optimizer_type == _OptimizerType.ADAMW:
        weight_decay_const_value = 1e-2

    _AttrReader(
        attr_readers, "weight_decay", _GroupGetter(),
        _ValueConstPairFormatter(variable_attrs,
                                 _IsEqualTo(weight_decay_const_value)))
    _AttrReader(attr_readers, "momentum", _GroupGetter(),
                _ValueConstPairFormatter(variable_attrs, _IsEqualTo(0.0)))
    _AttrReader(attr_readers, "velocity_scaling", _GroupGetter(1.0),
                _ValueConstPairFormatter(variable_attrs, _IsEqualTo(1.0)))
    _AttrReader(attr_readers, "dampening", _GroupGetter(),
                _ValueConstPairFormatter(variable_attrs, _IsEqualTo(0.0)))
    _AttrReader(attr_readers, "eps", _GroupGetter(),
                _ValueConstPairFormatter(variable_attrs, _IsEqualTo(1e-08)))
    _AttrReader(attr_readers, "max_weight_norm", _GroupGetter(),
                _ValueConstPairFormatter(variable_attrs, _IsEqualTo(65500.0)))
    _AttrReader(attr_readers, "alpha", _GroupGetter(),
                _ValueConstPairFormatter(variable_attrs, isAlwaysConst))
    _BetaReader(attr_readers, variable_attrs)

    # Split the optimizer's attributes in one of the three categories:
    # - Group variables
    # - Optimizer variables
    # - Optimizer attributes
    #
    # The optimizer dictionary we send to the backend is structured like:
    # {
    #   "optimizer_type": type,
    #   "opt_attrs_0": value,
    #   ...
    #   "defaults": {
    #       "group_vars_0": (value, is_const),
    #       ...
    #       "opt_vars_0": (value, is_const),
    #       ...
    #   },
    #   "groups": [
    #       {
    #           "group_vars_0": (value, is_const),
    #           ...
    #       },
    #       ...
    #   ]
    # }
    group_vars = opt_class._group_vars  # pylint: disable=protected-access
    all_attrs = [
        attr for attr in opt_class._child_only if attr not in group_vars  # pylint: disable=protected-access
    ]
    opt_attrs = [
        attr for attr in all_attrs if attr not in opt_class._child_vars  # pylint: disable=protected-access
    ]
    opt_vars = [
        attr for attr in opt_class._child_only  # pylint: disable=protected-access
        if attr in opt_class._child_vars  # pylint: disable=protected-access
    ]

    def getOptimizerAttrNames(opt):
        # Remove attributes belonging to the upstream Optimizer
        exceptions = ["defaults", "state", "param_groups", "variable_attrs"]
        return [k for k in opt.__dict__.keys() if k not in exceptions]

    def getGroupAttrNames(group):
        # Remove attributes belonging to the upstream Optimizer
        exceptions = ["params"]
        return [k for k in group.keys() if k not in exceptions]

    opts = {"optimizer_type": optimizer_type}
    for attr in opt_attrs:
        opts.update(attr_readers[attr](optimizer))
    defaults = {}
    for attr in group_vars:
        defaults.update(attr_readers[attr](optimizer.defaults))
    attr_tracker.checkDefaultAttributes(list(optimizer.defaults.keys()))
    for attr in opt_vars:
        defaults.update(attr_readers[attr](optimizer))
    attr_tracker.checkOptimizerAttributes(getOptimizerAttrNames(optimizer))
    for i, g in enumerate(optimizer.param_groups):
        attr_tracker.checkGroupAttributes(getGroupAttrNames(g), i)

    opts["defaults"] = defaults

    # Create num_groups dictionaries
    opts["groups"] = []
    for index in range(0, num_groups):
        group = {}
        params = optimizer.param_groups[index]
        for attr in group_vars:
            group.update(attr_readers[attr](params))
        opts["groups"].append(group)

    logger.debug("Python optimizer %s", opts)
    # From now on print a message when encountering unknown attributes
    attr_tracker.enableChecks()
    return opts


class _OptimizerType(enum.IntEnum):
    SGD1 = 0
    SGD2 = 1
    ADAM = 2
    ADAMW = 3
    ADAMW_NO_BIAS = 4
    RMSPROP = 5
    RMSPROP_CENTERED = 6
    LAMB = 7
    LAMB_NO_BIAS = 8


def _toPoptorchClass(optimizer_type):
    assert isinstance(optimizer_type, _OptimizerType)
    if optimizer_type in [_OptimizerType.ADAMW, _OptimizerType.ADAMW_NO_BIAS]:
        return optim.AdamW
    if optimizer_type in [
            _OptimizerType.RMSPROP, _OptimizerType.RMSPROP_CENTERED
    ]:
        return optim.RMSprop
    if optimizer_type in [_OptimizerType.LAMB, _OptimizerType.LAMB_NO_BIAS]:
        return optim.LAMB
    if optimizer_type in [_OptimizerType.SGD1, _OptimizerType.SGD2]:
        return optim.SGD
    assert optimizer_type == _OptimizerType.ADAM, (
        "Unknown optimizer_type %s" % optimizer_type)
    return optim.Adam


# pylint: disable=too-many-return-statements
def _toPoptorchOptimizer(optimizer):
    if isinstance(optimizer, torch.optim.SGD):
        use_combined_accum = getattr(optimizer, "use_combined_accum", False)
        if use_combined_accum:
            return _OptimizerType.SGD1
        return _OptimizerType.SGD2

    if isinstance(optimizer, torch.optim.Adam):
        return _OptimizerType.ADAM

    if isinstance(optimizer, torch.optim.AdamW):
        if isinstance(optimizer, optim.AdamW):
            bias_correction = getattr(optimizer, "bias_correction", True)
            if not bias_correction:
                return _OptimizerType.ADAMW_NO_BIAS
        return _OptimizerType.ADAMW

    if isinstance(optimizer, torch.optim.RMSprop):
        centered = optimizer.param_groups[0]["centered"]
        for i, group in enumerate(optimizer.param_groups):
            assert group["centered"] == centered, (
                "All parameter groups must "
                "have the same value for the 'centered' attribute (Group 0: "
                f"{centered} / Group {i}: {group['centered']})")

        if centered:
            return _OptimizerType.RMSPROP_CENTERED
        return _OptimizerType.RMSPROP

    if isinstance(optimizer, optim.LAMB):
        bias_correction = getattr(optimizer, "bias_correction", True)
        if bias_correction:
            return _OptimizerType.LAMB
        return _OptimizerType.LAMB_NO_BIAS
    return None


def _toCamelCase(string):
    """Convert a snake case string (PyTorch) to camel case (PopART)"""
    words = string.split("_")
    return words[0] + "".join(w.capitalize() for w in words[1:])


class _GroupGetter:
    """Functor to access a parameter group attribute"""

    def __init__(self, default_value=None):
        self.default_value = default_value

    def __call__(self, group, name):
        assert isinstance(group, dict), (f"{name} must be stored in "
                                         "param_groups")
        value = group.get(name, self.default_value)
        assert value is not None, (f"Mandatory attribute {name} not found "
                                   "in optimizer group")
        return value


class _OptimizerGetter:
    """Functor to access an Optimizer attribute"""

    def __init__(self, default_value=None):
        self.default_value = default_value

    def __call__(self, opt, name):
        assert isinstance(opt, torch.optim.Optimizer), (
            f"{name} must be stored "
            "as an Optimizer attribute (Not in a group)")
        value = getattr(opt, name, self.default_value)
        assert value is not None, (f"Mandatory attribute {name} not found "
                                   "in optimizer attributes")
        return value


def _assertIsNumber(value, name):
    assert isinstance(value, numbers.Number), (f"Expected a number for {name}"
                                               f" but got {value} instead")


class _ValueConstPairFormatter:
    """Functor to format a value into a pair ``(value, is_const)`` where
    "is_const" is a boolean

    If ``variable_attrs`` is provided it will be used to determine the
    attribute's constness.

    Otherwise the const_evaluator function will be called.
    """

    def __init__(self, variable_attrs, const_evaluator, value_validator=None):
        assert variable_attrs is None or isinstance(variable_attrs,
                                                    optim.VariableAttributes)
        if value_validator is None:
            value_validator = _assertIsNumber
        self.value_validator = value_validator
        self.variable_attrs = variable_attrs
        self.const_evaluator = const_evaluator

    def __call__(self, value, name):
        self.value_validator(value, name)
        if self.variable_attrs:
            is_const = self.variable_attrs.isConstant(name)
        else:
            is_const = self.const_evaluator(value)
        return (value, is_const)


class _IsEqualTo:
    """Functor which returns True if the passed value is equal to the reference"""

    def __init__(self, reference):
        self.reference = reference

    def __call__(self, value):
        return value == self.reference


class _AttrReader:
    def __init__(self, readers, name, getter, formatter=None, new_name=None):
        if new_name is None:
            new_name = _toCamelCase(name)
        if formatter is None:
            formatter = lambda x, _: x

        self.name = name
        self.getter = getter
        self.new_name = new_name
        self.formatter = formatter

        # Register itself
        readers[name] = self

    def __call__(self, params):
        """Get the ``name`` attribute value from ``params`` (An ``optimizer`` or
           ``param_group``)
        - if ``name`` is not part of ``params`` then ``default_value`` will be
          used.
        - If no ``variable_attrs`` list and no const value are provided then
          only ``{name: value}`` will be returned.
        - if a ``variable_attrs`` object is provided then the parameter's
          constness will depend on whether or not it's marked as const.
        - if no list is provided but the parameter's value is equal to
          ``is_const_val`` then the parameter will be considered constant
        """
        value = self.getter(params, self.name)
        return {self.new_name: self.formatter(value, self.name)}


class _BetaReader(_AttrReader):
    def __init__(self, attr_readers, variable_attrs):
        def isAlwaysConst(_value):
            return True

        def assertIsFloatPair(value, name):
            assert isinstance(value, tuple) and len(value) == 2, (
                f"Expected a pair for {name}"
                f" but got {value} instead")
            _assertIsNumber(value[0], name + "[0]")
            _assertIsNumber(value[1], name + "[1]")

        super().__init__(
            attr_readers, "betas", _GroupGetter(),
            _ValueConstPairFormatter(variable_attrs, isAlwaysConst,
                                     assertIsFloatPair))

    def __call__(self, params):
        betas = super().__call__(params)["betas"]
        assert betas and isinstance(betas, tuple) and len(betas) == 2
        assert isinstance(betas[0], tuple) and len(
            betas[0]) == 2, ("'betas' group attribute must be a pair")
        return {
            "beta1": (betas[0][0], betas[1]),
            "beta2": (betas[0][1], betas[1])
        }
