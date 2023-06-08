# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import math
import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
import torch

from ._logging import logger


class VariableAttributes:
    """Track which attributes are variable or constant.

    Is accessible via any PopTorch optimizer via the ``variable_attrs``
    attribute.

    >>> opt = poptorch.optim.SGD(params, lr=0.01)
    >>> opt.variable_attrs.isConstant("lr")
    """

    def __init__(self, variable_attributes: List[str],
                 allowed_attributes: List[str]) -> None:
        """
        :param variable_attributes: list of variable attributes.
        :param allowed_attributes: list of all the attributes.
        """
        self._variable_attributes = variable_attributes
        self._allowed_attributes = allowed_attributes

    def isConstant(self, attr: str) -> bool:
        """Return True if the attribute is marked as constant"""
        return attr not in self._variable_attributes

    def markAsConstant(self, attr: str) -> None:
        """Explicitly mark an attribute as constant"""
        assert attr in self._allowed_attributes, (
            f"Unknown attribute {attr},"
            f" allowed values: {self._allowed_attributes}")
        self._variable_attributes = [
            a for a in self._variable_attributes if a != attr
        ]

    def markAsVariable(self, attr: str) -> None:
        "Explicitly mark an attribute as variable" ""
        assert attr in self._allowed_attributes, (
            f"Unknown attribute {attr},"
            f" allowed values: {self._allowed_attributes}")
        self._variable_attributes.append(attr)


def _parseArgs(all_args: Dict[str, Any],
               child_attrs: Optional[List[str]] = None
               ) -> Tuple[Dict[str, Any], List[str]]:
    child_attrs = child_attrs or []
    args = all_args.copy()
    # Remove special local() variables
    del args["self"]
    # Attributes explicitly set by the user are considered variable
    not_const = [k for k, v in args.items() if v is not None]
    # Filter out the child class attributes
    parent_args = {
        k: v
        for k, v in args.items() if k in not_const and k not in child_attrs
    }
    return parent_args, not_const


class Optimizer:
    def __init__(self):
        self._state_dict = {"ipu_state": None, "ipu_param": None}
        # If True then the state needs to be uploaded to the IPU.
        self.ipu_state_is_dirty = False
        # Once the optimizer has been used on the IPU its state
        # on the host will become dirty.
        self.host_state_is_dirty = False

    # These functions must be overridden so that the optimiser state can be set
    # when the model is created
    def state_dict(self):
        return self.get_state_dict()

    def load_state_dict(self, state):
        # We also need to load torch's state dict so that LR schedulers work
        torch.optim.Optimizer.load_state_dict(self, state)
        self.set_state_dict(state)

    # Getter/setter for local state dict after the above functions been overridden by PoplarExecutor
    def get_state_dict(self):
        # Return both the internal state dict and torch's state dict
        # so that LR schedulers work
        return {**self._state_dict, **torch.optim.Optimizer.state_dict(self)}

    def set_state_dict(self, state):
        if not state:
            raise RuntimeError(
                "Cannot load optimizer state dictionary because it is empty.")
        if not ("ipu_state" in state and "ipu_param" in state):
            raise RuntimeError(
                "Only IPU optimizer states can be loaded onto the IPU.")
        self._state_dict = state
        self.ipu_state_is_dirty = True
        self.host_state_is_dirty = False

    def has_state(self):
        return (self._state_dict.get("ipu_state") is not None
                and self._state_dict.get("ipu_param") is not None)


class SGD(Optimizer, torch.optim.SGD):
    # pylint: disable=line-too-long
    """ Stochastic gradient descent with optional momentum.

    The optimizer is based on PyTorch's implementation
    (`torch.optim.SGD <https://pytorch.org/docs/1.10.0/optim.html#torch.optim.SGD>`_)
    with optional loss and velocity scaling.

    PopTorch provides two possible variants. Both variants are mathematically
    identical to PyTorch but differ in their stability and efficiency.

    .. note:: If you set momentum to zero and do not use gradient accumulation,
      PopTorch will use a simple SGD variant and ignore the values of
      ``use_combined_accum``, ``accum_type`` and ``velocity_accum_type``.

    **Separate tensor variant (default)**

    If you set ``use_combined_accum`` to ``False`` (default), you will use a
    more stable but more memory intensive variant. In this case, PopTorch keeps
    two state tensors for each weight: one for gradient accumulation and one for
    velocity. It operates as follows when training:

    #. PopTorch runs one or more forward/backwards steps, equal the number of
       gradient accumulations (see
       :py:func:`~poptorch.options._TrainingOptions.gradientAccumulation`).
       Each time PopTorch sums the gradients, storing them in accumulators.
    #. Once all the forward and backwards have completed, PopTorch uses the
       summed gradients to update the velocities. At this stage, PopTorch will
       correct the scale based on the setting of
       :py:func:`~poptorch.options._TrainingOptions.accumulationAndReplicationReductionType`.
       PopTorch stores the velocities as optimiser states.
    #. Finally, PopTorch uses the velocities to update the parameters, taking
       into account the loss scaling and learning rate.

    With ``use_combined_accum`` set to False, you can independently change the
    data type used for storing the accumulated gradients and the velocity
    values using ``accum_type`` and ``velocity_accum_type``, respectively.

    Velocity scaling is ignored for this variant.

    .. note:: If the number of gradient accumulations is high, you can use off
        chip memory for the velocity tensors with a minimal performance hit.

        >>> opts.TensorLocations.setOptimizerLocation(
        ...     poptorch.TensorLocationSettings().useOnChipStorage(False))

    **Combined tensor variant**

    If you set `use_combined_accum`` to ``True``, you will use a less stable but
    more memory efficient variant. In this case PopTorch uses a single tensor
    (the combined tensor) for gradient accumulation and velocity.
    It operates as follows when training:

    #. PopTorch runs one or more forward/backwards steps equal the number of
       gradient accumulations (see
       :py:func:`~poptorch.options._TrainingOptions.gradientAccumulation`).
       For each step, PopTorch immediately calculates an increment or decrement
       for the combined tensors for each parameter. The amount of increment or
       decrement takes into account the setting of
       :py:func:`~poptorch.options._TrainingOptions.accumulationAndReplicationReductionType`.
       as well as removing loss scaling and introducing any velocity scaling.
    #. After running all the steps, the combined tensor will be be equal to the
       new velocities. PopTorch uses these to update the parameters taking
       into account the velocity scaling and learning rate.

    PopTorch ignores the `accum_type`` and ``velocity_accum_type`` values when
    using a combined tensor. In addition, there are no optimizer state tensors
    and so ``opts.TensorLocations.setOptimizerLocation`` has no effect.

    .. warning:: For both variants, reducing the velocity scaling during
        training will result in temporary over-estimation of the velocity and
        could cause model instability. Increasing the scaling may temporarily
        slow model convergence but not lead to instability.
    """
    # Variables which don't exist in the parent optimizer class and are
    # global (Cannot be set per group).
    _child_vars = ["loss_scaling"]
    # All the attributes and variables which don't exist in the parent optimizer class.
    _child_only = _child_vars + [
        "velocity_scaling", "use_combined_accum", "accum_type",
        "velocity_accum_type", "max_grad_norm"
    ]
    # Attributes (from the parent or child class) which can be set per group.
    _group_vars = [
        "lr", "momentum", "dampening", "weight_decay", "nesterov",
        "velocity_scaling"
    ]

    def __init__(self,
                 params: Iterable,
                 lr: float,
                 momentum: Optional[float] = None,
                 dampening: Optional[float] = None,
                 weight_decay: Optional[float] = None,
                 nesterov: Optional[bool] = None,
                 maximize: Optional[bool] = None,
                 foreach: Optional[bool] = None,
                 differentiable: Optional[bool] = None,
                 loss_scaling: Optional[float] = None,
                 velocity_scaling: Optional[float] = None,
                 use_combined_accum: Optional[bool] = None,
                 accum_type: Optional[torch.dtype] = None,
                 velocity_accum_type: Optional[torch.dtype] = None,
                 max_grad_norm: Optional[float] = None) -> None:
        """
        :param iterable params: parameters to optimize.
        :param lr: learning rate.
        :param momentum: momentum factor.
        :param dampening: dampening term for momentum.
        :param weight_decay: Weight decay (L2 penalty) factor.
        :param nesterov: Whether to enable Nesterov momentum. Default is
            `False`.
        :param loss_scaling: Factor by which to scale the loss and hence
            gradients to assist numerical stability when using float16.
        :param velocity_scaling: Factor by which to scale the velocity values
            to assist numerical stability when using float16. (This applies to
            the combined variant only.)
        :param use_combined_accum: Whether to use a combined accumulator.
        :param accum_type: data type used for gradients.
        :param velocity_accum_type: data type used to store
            the velocity values for each parameter.
        :param max_grad_norm: Maximum norm of gradients. Default is `inf`.
        """
        # pylint: disable=unused-argument
        # Call to locals() must be at the very top of  __init__
        parent_args, variables = _parseArgs(locals(), SGD._child_only)
        Optimizer.__init__(self)
        torch.optim.SGD.__init__(self, **parent_args)

        # Loss scaling is a global setting: store it as an attribute
        if loss_scaling is None:
            loss_scaling = 1.0

        if use_combined_accum is None:
            use_combined_accum = False
        self.use_combined_accum = use_combined_accum

        if accum_type is None:
            accum_type = torch.float32
        if velocity_accum_type is None:
            velocity_accum_type = torch.float32

        self.loss_scaling = loss_scaling

        # Velocity scaling can be set per group: register it in defaults
        # and update the existing groups.
        if velocity_scaling is None:
            velocity_scaling = 1.0
            # NB this will be overridden to loss_scaling in the case of the
            # separate tensor variant.
        else:
            if not use_combined_accum:
                logger.warning("velocity_scaling value ignored when "
                               "using the separate variant "
                               "(use_combined_accum=False). In future, this "
                               "will lead to an error. Please update your "
                               "code.")

        if use_combined_accum:
            self.defaults["velocity_scaling"] = velocity_scaling
            for group in self.param_groups:
                group.setdefault("velocity_scaling", velocity_scaling)

        if nesterov is None:
            nesterov = False

        supportedTypes = [torch.float16, torch.float32]
        errString = ("Accumulation types must be either torch.float32"
                     " or torch.float16")
        assert accum_type in supportedTypes, errString
        self.accum_type = accum_type

        assert velocity_accum_type in supportedTypes, errString
        self.velocity_accum_type = velocity_accum_type
        if max_grad_norm is None:
            max_grad_norm = float("Inf")
        self.max_grad_norm = max_grad_norm

        self.variable_attrs = VariableAttributes(
            variables,
            list(self.defaults) + SGD._child_vars)

    def __getstate__(self) -> Dict[str, Any]:
        state = torch.optim.SGD.__getstate__(self)
        # Manually save the attributes
        # (groups / defaults are saved by the parent)
        state["variable_attrs"] = self.variable_attrs
        state["loss_scaling"] = self.loss_scaling
        state["use_combined_accum"] = self.use_combined_accum
        state["accum_type"] = self.accum_type
        state["velocity_accum_type"] = self.velocity_accum_type
        state["max_grad_norm"] = self.max_grad_norm

        # Mark the state as dirty only if there is one.
        state["_state_dict"] = self._state_dict
        state["ipu_state_is_dirty"] = self.has_state()
        state["host_state_is_dirty"] = False
        return state


class Adam(Optimizer, torch.optim.Adam):
    """ Adam optimizer.

    This optimizer matches PyTorch's implementation
    (`torch.optim.Adam <https://pytorch.org/docs/1.10.0/optim.html#torch.optim.Adam>`_) with
    optional loss scaling.

    AMSGrad is currently not supported."""

    # Variables which don't exist in the parent optimizer class and are
    # global (Cannot be set per group).
    _child_vars = ["loss_scaling"]
    # All the attributes and variables which don't exist in the parent optimizer class.
    _child_only = _child_vars + [
        "accum_type", "first_order_momentum_accum_type",
        "second_order_momentum_accum_type", "max_grad_norm"
    ]
    # Attributes (from the parent or child class) which can be set per group.
    _group_vars = ["lr", "betas", "eps", "weight_decay", "amsgrad"]

    def __init__(
            self,
            params: Iterable,
            lr: Optional[float] = None,
            betas: Optional[Tuple[float, float]] = None,
            eps: Optional[float] = None,
            weight_decay: Optional[float] = None,
            amsgrad: Optional[bool] = None,
            foreach: Optional[bool] = None,
            maximize: Optional[bool] = None,
            capturable: Optional[bool] = None,
            differentiable: Optional[bool] = None,
            fused: Optional[bool] = None,
            loss_scaling: Optional[float] = None,
            accum_type: Optional[torch.dtype] = None,
            first_order_momentum_accum_type: Optional[torch.dtype] = None,
            second_order_momentum_accum_type: Optional[torch.dtype] = None,
            max_grad_norm: Optional[float] = None) -> None:
        """
        :param iterable params: parameters to optimize.
        :param lr: learning rate
        :param betas: ``(beta1, beta2)`` parameters used in Adam.
        :param eps: term added to the denominator to ensure numerical stability.
        :param weight_decay: Weight decay factor.
        :param amsgrad: Not supported (must be False).
        :param loss_scaling: Factor by which to scale the loss and hence
            gradients to assist numerical stability when using float16.
        :param accum_type: data type used for gradients.
        :param first_order_momentum_accum_type: data type used to store
            the first order momentum values for each parameter.
        :param second_order_momentum_accum_type: data type used to store
            the second order momentum values for each parameter.
        :param max_grad_norm: Maximum norm of gradients. Default is `inf`.
        """
        # pylint: disable=unused-argument
        # Call to locals() must be at the very top of  __init__
        parent_args, variables = _parseArgs(locals(), Adam._child_only)
        Optimizer.__init__(self)
        torch.optim.Adam.__init__(self, **parent_args)

        if loss_scaling is None:
            loss_scaling = 1.0
        if accum_type is None:
            accum_type = torch.float32
        if first_order_momentum_accum_type is None:
            first_order_momentum_accum_type = torch.float32
        if second_order_momentum_accum_type is None:
            second_order_momentum_accum_type = torch.float32
        if max_grad_norm is None:
            max_grad_norm = float("Inf")

        # All the child attributes are global: store them as
        # attributes.
        self.loss_scaling = loss_scaling

        supportedTypes = [torch.float16, torch.float32]
        errString = ("Accumulation types must be either torch.float32"
                     " or torch.float16")
        assert accum_type in supportedTypes, errString
        self.accum_type = accum_type

        assert first_order_momentum_accum_type in supportedTypes, errString
        self.first_order_momentum_accum_type = \
             first_order_momentum_accum_type

        assert second_order_momentum_accum_type in supportedTypes, errString
        self.second_order_momentum_accum_type = \
             second_order_momentum_accum_type

        self.max_grad_norm = max_grad_norm

        self.variable_attrs = VariableAttributes(
            variables,
            list(self.defaults) + Adam._child_vars)

    def __getstate__(self) -> Dict[str, Any]:
        state = torch.optim.Adam.__getstate__(self)
        # Manually save the attributes
        # (groups / defaults are saved by the parent)
        state["variable_attrs"] = self.variable_attrs
        state["loss_scaling"] = self.loss_scaling
        state["accum_type"] = self.accum_type
        state["first_order_momentum_accum_type"] = \
                self.first_order_momentum_accum_type
        state["second_order_momentum_accum_type"] = \
                self.second_order_momentum_accum_type
        state["max_grad_norm"] = self.max_grad_norm

        # Mark the state as dirty only if there is one.
        state["_state_dict"] = self._state_dict
        state["ipu_state_is_dirty"] = self.has_state()
        state["host_state_is_dirty"] = False
        return state


class AdamW(Optimizer, torch.optim.AdamW):
    """ Adam optimizer with true weight decay.

    This optimizer matches PyTorch's implementation
    (`torch.optim.AdamW <https://pytorch.org/docs/1.10.0/optim.html#torch.optim.AdamW>`_)
    with optional loss scaling.

    AMSGrad is currently not supported."""

    # Variables which don't exist in the parent optimizer class and are
    # global (Cannot be set per group).
    _child_vars = ["loss_scaling"]
    # All the attributes and variables which don't exist in the parent optimizer class.
    _child_only = _child_vars + [
        "bias_correction",
        "accum_type",
        "first_order_momentum_accum_type",
        "second_order_momentum_accum_type",
        "max_grad_norm",
    ]
    # Attributes (from the parent or child class) which can be set per group.
    _group_vars = ["lr", "betas", "weight_decay", "eps", "amsgrad"]

    def __init__(
            self,
            params: Iterable,
            lr: Optional[float] = None,
            betas: Optional[Tuple[float, float]] = None,
            eps: Optional[float] = None,
            weight_decay: Optional[float] = None,
            amsgrad: Optional[bool] = None,
            maximize: Optional[bool] = None,
            foreach: Optional[bool] = None,
            capturable: Optional[bool] = None,
            differentiable: Optional[bool] = None,
            fused: Optional[bool] = None,
            loss_scaling: Optional[float] = None,
            bias_correction: Optional[bool] = None,
            accum_type: Optional[torch.dtype] = None,
            first_order_momentum_accum_type: Optional[torch.dtype] = None,
            second_order_momentum_accum_type: Optional[torch.dtype] = None,
            max_grad_norm: Optional[float] = None) -> None:
        """
        :param iterable params: parameters to optimize.
        :param lr: learning rate
        :param betas: ``(beta1, beta2)`` parameters used in AdamW.
        :param eps: term added to the denominator to ensure numerical stability.
        :param weight_decay: Weight decay factor.
        :param amsgrad: Not supported (must be False).
        :param loss_scaling: Factor by which to scale the loss and hence
            gradients to assist numerical stability when using float16.
        :param bias_correction: True: compute Adam with bias correction.
        :param accum_type: data type used for gradients.
        :param first_order_momentum_accum_type: data type used to store
            the first order momentum values for each parameter.
        :param second_order_momentum_accum_type: data type used to store
            the second order momentum values for each parameter.
        :param max_grad_norm: Maximum norm of gradients. Default is `inf`.
        """
        # pylint: disable=unused-argument
        # Call to locals() must be at the very top of  __init__
        parent_args, variables = _parseArgs(locals(), AdamW._child_only)
        Optimizer.__init__(self)
        torch.optim.AdamW.__init__(self, **parent_args)

        if loss_scaling is None:
            loss_scaling = 1.0
        if bias_correction is None:
            bias_correction = True
        if accum_type is None:
            accum_type = torch.float32
        if first_order_momentum_accum_type is None:
            first_order_momentum_accum_type = torch.float32
        if second_order_momentum_accum_type is None:
            second_order_momentum_accum_type = torch.float32
        if max_grad_norm is None:
            max_grad_norm = float("Inf")

        self.loss_scaling = loss_scaling
        self.bias_correction = bias_correction

        supportedTypes = [torch.float16, torch.float32]
        errString = ("Accumulation types must be either torch.float32"
                     " or torch.float16")
        assert accum_type in supportedTypes, errString
        self.accum_type = accum_type

        assert first_order_momentum_accum_type in supportedTypes, errString
        self.first_order_momentum_accum_type = \
             first_order_momentum_accum_type

        assert second_order_momentum_accum_type in supportedTypes, errString
        self.second_order_momentum_accum_type = \
             second_order_momentum_accum_type

        self.max_grad_norm = max_grad_norm

        self.variable_attrs = VariableAttributes(
            variables,
            list(self.defaults) + AdamW._child_vars)

    def __getstate__(self) -> Dict[str, Any]:
        state = torch.optim.AdamW.__getstate__(self)
        # Manually save the attributes
        # (groups / defaults are saved by the parent)
        state["variable_attrs"] = self.variable_attrs
        state["loss_scaling"] = self.loss_scaling
        state["bias_correction"] = self.bias_correction
        state["accum_type"] = self.accum_type
        state["first_order_momentum_accum_type"] = \
                self.first_order_momentum_accum_type
        state["second_order_momentum_accum_type"] = \
                self.second_order_momentum_accum_type
        state["max_grad_norm"] = self.max_grad_norm

        # Mark the state as dirty only if there is one.
        state["_state_dict"] = self._state_dict
        state["ipu_state_is_dirty"] = self.has_state()
        state["host_state_is_dirty"] = False
        return state


class RMSprop(Optimizer, torch.optim.RMSprop):
    """ RMSprop optimizer with optional L2 penalty.

    This optimizer matches PyTorch's implementation (
    `torch.optim.RMSprop <https://pytorch.org/docs/1.10.0/optim.html#torch.optim.RMSprop>`_)
    with optional loss scaling.

    However, if the use_tf_variant flag is set to True, it will instead match
    the TensorFlow implementation which differs from PyTorch's implementation
    in three ways:
    1) The average squared gradients buffer is initialized to ones.
    2) The small epsilon constant is applied inside the square root.
    3) Learning rate is accumulated in the momentum buffer if momentum is used."""

    # Variables which don't exist in the parent optimizer class and are
    # global (Cannot be set per group).
    _child_vars = ["loss_scaling"]
    # All the attributes and variables which don't exist in the parent optimizer class.
    _child_only = _child_vars + [
        "accum_type", "first_order_momentum_accum_type",
        "second_order_momentum_accum_type", "use_tf_variant"
    ]
    # Attributes (from the parent or child class) which can be set per group.
    _group_vars = [
        "lr", "momentum", "weight_decay", "alpha", "eps", "centered"
    ]

    def __init__(
            self,
            params: Iterable,
            lr: Optional[float] = None,
            alpha: Optional[float] = None,
            eps: Optional[float] = None,
            weight_decay: Optional[float] = None,
            momentum: Optional[float] = None,
            centered: Optional[bool] = None,
            foreach: Optional[bool] = None,
            maximize: Optional[bool] = None,
            differentiable: Optional[bool] = None,
            loss_scaling: Optional[float] = None,
            accum_type: Optional[torch.dtype] = None,
            first_order_momentum_accum_type: Optional[torch.dtype] = None,
            second_order_momentum_accum_type: Optional[torch.dtype] = None,
            use_tf_variant: Optional[bool] = None) -> None:
        """
        :param iterable params: parameters to optimize.
        :param lr: learning rate.
        :param alpha: smoothing constant.
        :param eps: term added to the denominator to ensure numerical
           stability.
        :param weight_decay: L2 penalty coefficient.
        :param momentum: momentum factor.
        :param centered: True: compute centred RMSprop in which the
            gradient is normalized by an estimate of its variance.
        :param loss_scaling: Factor by which to scale the loss and hence
            gradients to assist numerical stability when using float16.
        :param accum_type: data type used for gradients.
        :param first_order_momentum_accum_type: data type used to store
            the first order momentum values for each parameter.
        :param second_order_momentum_accum_type: data type used to store
            the second order momentum values for each parameter.
        :param use_tf_variant: False: If True, use the TensorFlow variant
            of RMSProp.
        """
        # pylint: disable=unused-argument
        # Call to locals() must be at the very top of  __init__
        parent_args, variables = _parseArgs(locals(), RMSprop._child_only)
        Optimizer.__init__(self)
        torch.optim.RMSprop.__init__(self, **parent_args)

        if loss_scaling is None:
            loss_scaling = 1.0
        if accum_type is None:
            accum_type = torch.float32
        if first_order_momentum_accum_type is None:
            first_order_momentum_accum_type = torch.float32
        if second_order_momentum_accum_type is None:
            second_order_momentum_accum_type = torch.float32
        if use_tf_variant is None:
            use_tf_variant = False

        self.loss_scaling = loss_scaling

        supportedTypes = [torch.float16, torch.float32]
        errString = ("Accumulation types must be either torch.float32"
                     " or torch.float16")
        assert accum_type in supportedTypes, errString
        self.accum_type = accum_type

        assert first_order_momentum_accum_type in supportedTypes, errString
        self.first_order_momentum_accum_type = \
             first_order_momentum_accum_type

        assert second_order_momentum_accum_type in supportedTypes, errString
        self.second_order_momentum_accum_type = \
             second_order_momentum_accum_type
        self.use_tf_variant = use_tf_variant
        self.variable_attrs = VariableAttributes(
            variables,
            list(self.defaults) + RMSprop._child_vars)

    def __getstate__(self) -> Dict[str, Any]:
        state = torch.optim.RMSprop.__getstate__(self)
        # Manually save the attributes
        # (groups / defaults are saved by the parent)
        state["variable_attrs"] = self.variable_attrs
        state["loss_scaling"] = self.loss_scaling
        state["accum_type"] = self.accum_type
        state["first_order_momentum_accum_type"] = \
                self.first_order_momentum_accum_type
        state["second_order_momentum_accum_type"] = \
                self.second_order_momentum_accum_type
        state["use_tf_variant"] = self.use_tf_variant

        # Mark the state as dirty only if there is one.
        state["_state_dict"] = self._state_dict
        state["ipu_state_is_dirty"] = self.has_state()
        state["host_state_is_dirty"] = False
        return state


class LAMB(Optimizer, torch.optim.Optimizer):
    """ Layer-wise Adaptive Moments (LAMB) optimizer (biased version).

        Based on "Large Batch Optimization for Deep Learning: Training BERT
        in 76 minutes" (https://arxiv.org/abs/1904.00962).

        The scaling function phi(z) is fixed as min(z, max_weight_norm);
    """
    # Variables which don't exist in the parent optimizer class and are
    # global (Cannot be set per group).
    _child_vars = ["loss_scaling"]
    # All the attributes and variables which don't exist in the parent optimizer class.
    _child_only = _child_vars + [
        "bias_correction", "accum_type", "first_order_momentum_accum_type",
        "second_order_momentum_accum_type"
    ]
    # Attributes (from the parent or child class) which can be set per group.
    _group_vars = ["lr", "weight_decay", "betas", "eps", "max_weight_norm"]

    def __init__(self,
                 params: Iterable,
                 lr: Optional[float] = None,
                 betas: Tuple[float, float] = None,
                 eps: Optional[float] = None,
                 weight_decay: Optional[float] = None,
                 bias_correction: Optional[bool] = None,
                 loss_scaling: Optional[float] = None,
                 max_weight_norm: Optional[float] = None,
                 accum_type: Optional[torch.dtype] = None,
                 first_order_momentum_accum_type: Optional[torch.dtype] = None,
                 second_order_momentum_accum_type: Optional[torch.dtype] = None
                 ) -> None:
        """
        :param iterable params: parameters to optimize.
        :param lr: learning rate
        :param betas: ``(beta1, beta2)`` parameters used in LAMB.
        :param eps: term added to the denominator to ensure numerical
           stability/
        :param weight_decay: weight decay factor.
        :param bias_correction: True: compute LAMB with bias correction.
        :param loss_scaling: Factor by which to scale the loss and hence
            gradients to assist numerical stability when using float16.
        :param max_weight_norm: maximum value of the output of scaling
            function, phi(). Set to None to disable scaling function.
        :param accum_type: data type used for gradients.
        :param first_order_momentum_accum_type: data type used to store
            the first order momentum values for each parameter.
        :param second_order_momentum_accum_type: data type used to store
           the second order momentum values for each parameter.
        """
        # pylint: disable=unused-argument
        # Call to locals() must be at the very top of  __init__
        _, variables = _parseArgs(locals(), [])
        if max_weight_norm is None:
            max_weight_norm = 65500.0  # FP16 Max
        if lr is None:
            lr = 1e-3
        if betas is None:
            betas = (0.9, 0.999)
        if eps is None:
            eps = 1e-8
        if weight_decay is None:
            weight_decay = 1e-2
        if bias_correction is None:
            bias_correction = True
        if loss_scaling is None:
            loss_scaling = 1.0
        if accum_type is None:
            accum_type = torch.float32
        if first_order_momentum_accum_type is None:
            first_order_momentum_accum_type = torch.float32
        if second_order_momentum_accum_type is None:
            second_order_momentum_accum_type = torch.float32
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        max_weight_norm=max_weight_norm)
        Optimizer.__init__(self)
        torch.optim.Optimizer.__init__(self, params, defaults)

        supportedTypes = [torch.float16, torch.float32]
        errString = """Accumulation types must be either torch.float32
                or torch.float16"""
        assert accum_type in supportedTypes, errString
        assert first_order_momentum_accum_type in supportedTypes, errString
        assert second_order_momentum_accum_type in supportedTypes, errString

        self.bias_correction = bias_correction
        self.loss_scaling = loss_scaling
        self.max_weight_norm = max_weight_norm
        self.accum_type = accum_type
        self.first_order_momentum_accum_type = \
             first_order_momentum_accum_type
        self.second_order_momentum_accum_type = \
             second_order_momentum_accum_type

        self.variable_attrs = VariableAttributes(
            variables,
            list(self.defaults) + LAMB._child_vars)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                beta1, beta2 = group["betas"]

                if self.bias_correction:
                    bias_correction1 = 1 - beta1**state["step"]
                    bias_correction2 = 1 - beta2**state["step"]
                else:
                    bias_correction1 = 1
                    bias_correction2 = 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"])

                upd = ((exp_avg / bias_correction1) /
                       denom) + group["weight_decay"] * p.data

                r1 = p.data.pow(2).sum().sqrt()
                r2 = upd.pow(2).sum().sqrt()

                r1_ = r1.clamp(max=self.max_weight_norm)

                if r1_ == 0 or r2 == 0:
                    trust = 1.0
                else:
                    trust = r1_ / r2

                p.data.add_(upd, alpha=-group['lr'] * trust)

        return loss

    def __getstate__(self) -> Dict[str, Any]:
        state = torch.optim.Optimizer.__getstate__(self)
        # Manually save the attributes
        # (groups / defaults are saved by the parent)
        state["variable_attrs"] = self.variable_attrs
        state["loss_scaling"] = self.loss_scaling
        state["bias_correction"] = self.bias_correction
        state["accum_type"] = self.accum_type
        state["first_order_momentum_accum_type"] = \
                self.first_order_momentum_accum_type
        state["second_order_momentum_accum_type"] = \
                self.second_order_momentum_accum_type

        # Mark the state as dirty only if there is one.
        state["_state_dict"] = self._state_dict
        state["ipu_state_is_dirty"] = self.has_state()
        state["host_state_is_dirty"] = False
        return state


def _check_constructor_match_parent(child_class: Type[torch.optim.Optimizer]
                                    ) -> None:
    parent = child_class.__bases__[1]
    parent_params = inspect.signature(parent.__init__).parameters
    child_params = inspect.signature(child_class.__init__).parameters
    extra_args = child_class._child_only  # pylint: disable=protected-access
    assert len(parent_params) + len(extra_args) == len(child_params), (
        f"Expected {len(parent_params) + len(extra_args)} parameters but got "
        f"{len(child_params)}")

    child_params = iter(child_params.items())
    for idx, (_, param) in enumerate(parent_params.items()):
        _, child_param = next(child_params)
        assert child_param.name == param.name, (
            f"Mismatch for parameter {idx}: expected"
            f"'{param}' but got '{child_param}'")

    for extra_arg in extra_args:
        name, _ = next(child_params)
        assert name == extra_arg, (f"Expected an extra argument named "
                                   f"'{extra_arg}' but got '{name}'")


_check_constructor_match_parent(SGD)
_check_constructor_match_parent(Adam)
_check_constructor_match_parent(AdamW)
_check_constructor_match_parent(RMSprop)
