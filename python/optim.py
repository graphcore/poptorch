# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import math
import inspect
import torch


class VariableAttributes:
    """Track which attributes are variable or constant.
    """

    def __init__(self, variable_attributes, allowed_attributes):
        """
        :param variable_attributes: list of variable attributes.
        :param allowed_attributes: list of all the attributes.
        """
        self._variable_attributes = variable_attributes
        self._allowed_attributes = allowed_attributes

    def isConstant(self, attr):
        return attr not in self._variable_attributes

    def markAsConstant(self, attr):
        "Explicitly mark an attribute as constant" ""
        assert attr in self._allowed_attributes, (
            f"Unknown attribute {attr},"
            f" allowed values: {self._allowed_attributes}")
        self._variable_attributes = [
            a for a in self._variable_attributes if a != attr
        ]

    def markAsVariable(self, attr):
        "Explicitly mark an attribute as variable" ""
        assert attr in self._allowed_attributes, (
            f"Unknown attribute {attr},"
            f" allowed values: {self._allowed_attributes}")
        self._variable_attributes.append(attr)


def _parseArgs(all_args, child_attrs=None):
    child_attrs = child_attrs or []
    args = all_args.copy()
    # Remove special local() variables
    del args["self"]
    del args["__class__"]
    # Attributes explicitly set by the user are considered variable
    not_const = [k for k, v in args.items() if v is not None]
    # Filter out the child class attributes
    parent_args = {
        k: v
        for k, v in args.items() if k in not_const and k not in child_attrs
    }
    return parent_args, not_const


class SGD(torch.optim.SGD):
    """ Stochastic gradient descent with optional momentum.

    The optimizer matches PyTorch's implementation (torch.optim.SGD) with
    optional loss and velocity scaling.

    Nesterov momentum is not currently supported.
    """
    # Variables which don't exist in the parent optimizer class and are
    # global (Cannot be set per group).
    _child_vars = ["loss_scaling"]
    # All the attributes and variables which don't exist in the parent optimizer class.
    _child_only = _child_vars + ["velocity_scaling"]
    # Attributes (from the parent or child class) which can be set per group.
    _group_vars = [
        "lr", "momentum", "dampening", "weight_decay", "nesterov",
        "velocity_scaling", "velocity_scaling", "nesterov"
    ]

    def __init__(self,
                 params,
                 lr=None,
                 momentum=None,
                 dampening=None,
                 weight_decay=None,
                 nesterov=None,
                 loss_scaling=None,
                 velocity_scaling=None):
        """
        :param iterable params: parameters to optimize.
        :param float lr: learning rate.
        :param float, optional momentum: momentum factor.
        :type momentum: float, optional
        :param dampening: damperning term for momentum.
        :type dampening: float, optional
        :param weight_decay: Weight decay (L2 penalty) factor.
        :type weight_decay: float, optional
        :param nesterov: Not supported (must be False).
        :type nesterov: bool, optional
        :param loss_scaling: Factor by which to scale the loss and hence
            gradients to assist numerical stability when using float16.
        :type loss_scaling: float, optional
        :param velocity_scaling: Factor by which to scale the velocity values
            to assist numerical stability when using float16.
        :type velocity_scaling: float, optional
        """
        # Call to locals() must be at the very top of  __init__
        parent_args, variables = _parseArgs(locals(), SGD._child_only)
        super().__init__(**parent_args)

        # Loss scaling is a global setting: store it as an attribute
        if loss_scaling is None:
            loss_scaling = 1.0
        self.loss_scaling = loss_scaling

        # Velocity scaling can be set per group: register it in defaults
        # and update the existing groups.
        if velocity_scaling is None:
            velocity_scaling = 1.0
        self.defaults["velocity_scaling"] = velocity_scaling
        for group in self.param_groups:
            group.setdefault("velocity_scaling", velocity_scaling)

        self.variable_attrs = VariableAttributes(
            variables,
            list(self.defaults) + SGD._child_vars)

    def __getstate__(self):
        state = super().__getstate__()
        # Manually save the attributes
        # (groups / defaults are saved by the parent)
        state["variable_attrs"] = self.variable_attrs
        state["loss_scaling"] = self.loss_scaling
        return state


class Adam(torch.optim.Adam):
    """ Adam optimizer.

    This optimizer matches PyTorch's implementation (torch.optim.Adam) with
    optional loss scaling.

    AMSGrad is currently not supported."""

    # Variables which don't exist in the parent optimizer class and are
    # global (Cannot be set per group).
    _child_vars = ["loss_scaling"]
    # All the attributes and variables which don't exist in the parent optimizer class.
    _child_only = _child_vars + [
        "accum_type", "first_order_momentum_accum_type",
        "second_order_momentum_accum_type"
    ]
    # Attributes (from the parent or child class) which can be set per group.
    _group_vars = ["lr", "betas", "eps", "weight_decay", "amsgrad"]

    def __init__(self,
                 params,
                 lr=None,
                 betas=None,
                 eps=None,
                 weight_decay=None,
                 amsgrad=None,
                 loss_scaling=None,
                 accum_type=None,
                 first_order_momentum_accum_type=None,
                 second_order_momentum_accum_type=None):
        """
        :param iterable params: parameters to optimize.
        :param lr: learning rate
        :type lr: float, optional
        :param betas: (beta1, beta2) parameters used in Adam.
        :type betas: tuple, optional
        :param eps: term added to the demoninator to ensure numerical stability.
        :type eps: float, optional
        :param weight_decay: Weight decay factor.
        :type weight_decay: float, optional
        :param amsgrad: Not supported (must be False).
        :type amsgrad: bool, optional
        :param loss_scaling: Factor by which to scale the loss and hence
            gradients to assist numerical stability when using float16.
        :type loss_scaling: float, optional
        :param accum_type: data type used for gradients.
        :type accum_type: torch.dtype, optional
        :param first_order_momentum_accum_type: data type used to store
            the first order momentum values for each parameter.
        :type first_order_momentum_accum_type: torch.dtype, optional
        :param second_order_momentum_accum_type: data type used to store
            the second order momentum values for each parameter.
        :type second_order_momentum_accum_type: torch.dtype, optional
        """
        # Call to locals() must be at the very top of  __init__
        parent_args, variables = _parseArgs(locals(), Adam._child_only)
        super().__init__(**parent_args)

        if loss_scaling is None:
            loss_scaling = 1.0
        if accum_type is None:
            accum_type = torch.float32
        if first_order_momentum_accum_type is None:
            first_order_momentum_accum_type = torch.float32
        if second_order_momentum_accum_type is None:
            second_order_momentum_accum_type = torch.float32

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

        self.variable_attrs = VariableAttributes(
            variables,
            list(self.defaults) + Adam._child_vars)

    def __getstate__(self):
        state = super().__getstate__()
        # Manually save the attributes
        # (groups / defaults are saved by the parent)
        state["variable_attrs"] = self.variable_attrs
        state["loss_scaling"] = self.loss_scaling
        state["accum_type"] = self.accum_type
        state["first_order_momentum_accum_type"] = \
                self.first_order_momentum_accum_type
        state["second_order_momentum_accum_type"] = \
                self.second_order_momentum_accum_type
        return state


class AdamW(torch.optim.AdamW):
    """ Adam optimizer with true weight decay.

    This optimizer matches PyTorch's implementation (torch.optim.AdamW) with
    optional loss scaling.

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
    ]
    # Attributes (from the parent or child class) which can be set per group.
    _group_vars = ["lr", "betas", "weight_decay", "eps", "amsgrad"]

    def __init__(self,
                 params,
                 lr=None,
                 betas=None,
                 eps=None,
                 weight_decay=None,
                 amsgrad=None,
                 loss_scaling=None,
                 bias_correction=None,
                 accum_type=None,
                 first_order_momentum_accum_type=None,
                 second_order_momentum_accum_type=None):
        """
        :param iterable params: parameters to optimize.
        :param lr: learning rate
        :type lr: float, optional
        :param betas: (beta1, beta2) parameters used in AdamW.
        :type betas: tuple, optional
        :param eps: term added to the demoninator to ensure numerical stability.
        :type eps: float, optional
        :param weight_decay: Weight decay factor.
        :type weight_decay: float, optional
        :param amsgrad: Not supported (must be False).
        :type amsgrad: bool, optional
        :param loss_scaling: Factor by which to scale the loss and hence
            gradients to assist numerical stability when using float16.
        :type loss_scaling: float, optional
        :param accum_type: data type used for gradients.
        :type accum_type: torch.dtype, optional
        :param first_order_momentum_accum_type: data type used to store
            the first order momentum values for each parameter.
        :type first_order_momentum_accum_type: torch.dtype, optional
        :param second_order_momentum_accum_type: data type used to store
            the second order momentum values for each parameter.
        :type second_order_momentum_accum_type: torch.dtype, optional
        """
        # Call to locals() must be at the very top of  __init__
        parent_args, variables = _parseArgs(locals(), AdamW._child_only)
        super().__init__(**parent_args)

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
        self.variable_attrs = VariableAttributes(
            variables,
            list(self.defaults) + AdamW._child_vars)

    def __getstate__(self):
        state = super().__getstate__()
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
        return state


class RMSprop(torch.optim.RMSprop):
    """ RMSprop optimizer with optional L2 penalty.

    This optimizer matches PyTorch's implementation (torch.optim.RMSprop) with
    optional loss scaling."""

    # Variables which don't exist in the parent optimizer class and are
    # global (Cannot be set per group).
    _child_vars = ["loss_scaling"]
    # All the attributes and variables which don't exist in the parent optimizer class.
    _child_only = _child_vars + [
        "accum_type", "first_order_momentum_accum_type",
        "second_order_momentum_accum_type"
    ]
    # Attributes (from the parent or child class) which can be set per group.
    _group_vars = [
        "lr", "momentum", "weight_decay", "alpha", "eps", "centered"
    ]

    def __init__(self,
                 params,
                 lr=None,
                 alpha=None,
                 eps=None,
                 weight_decay=None,
                 momentum=None,
                 centered=None,
                 loss_scaling=None,
                 accum_type=None,
                 first_order_momentum_accum_type=None,
                 second_order_momentum_accum_type=None):
        """
        :param iterable params: parameters to optimize.
        :param lr: learning rate.
        :type lr: float, optional
        :param alpha: smoothing constant.
        :type alpha: float, optional
        :param eps: term added to the demoninator to ensure numerical
           stability.
        :type eps: float, optional
        :param weight_decay: L2 penalty coeffecient.
        :type weight_decay: float, optional
        :param momentum: momentum factor.
        :type momentum: float, optional
        :param centered: True: compute centred RMSProp in which the
            gradient is normalized by an estimate of its variance.
        :type centered: bool, optional
        :param loss_scaling: Factor by which to scale the loss and hence
            gradients to assist numerical stability when using float16.
        :type loss_scaling: float, optional
        :param accum_type: data type used for gradients.
        :type accum_type: torch.dtype, optional
        :param first_order_momentum_accum_type: data type used to store
            the first order momentum values for each parameter.
        :type first_order_momentum_accum_type: torch.dtype, optional
        :param second_order_momentum_accum_type: data type used to store
            the second order momentum values for each parameter.
        :type second_order_momentum_accum_type: torch.dtype, optional
        """
        # Call to locals() must be at the very top of  __init__
        parent_args, variables = _parseArgs(locals(), RMSprop._child_only)
        super().__init__(**parent_args)

        if loss_scaling is None:
            loss_scaling = 1.0
        if accum_type is None:
            accum_type = torch.float32
        if first_order_momentum_accum_type is None:
            first_order_momentum_accum_type = torch.float32
        if second_order_momentum_accum_type is None:
            second_order_momentum_accum_type = torch.float32
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
        self.variable_attrs = VariableAttributes(
            variables,
            list(self.defaults) + RMSprop._child_vars)

    def __getstate__(self):
        state = super().__getstate__()
        # Manually save the attributes
        # (groups / defaults are saved by the parent)
        state["variable_attrs"] = self.variable_attrs
        state["loss_scaling"] = self.loss_scaling
        state["accum_type"] = self.accum_type
        state["first_order_momentum_accum_type"] = \
                self.first_order_momentum_accum_type
        state["second_order_momentum_accum_type"] = \
                self.second_order_momentum_accum_type
        return state


class LAMB(torch.optim.Optimizer):
    """ Layer-wise Adaptive Moments (LAMB) optimizer (biased version).

        Based on "Large Batch Optimization for Deep Learning: Training BERT
        in 76 minutes" (https://arxiv.org/abs/1904.00962).

        The scaling function phi(z) is fixed as min(z, max_weight_norm);
    """
    # Variables which don't exist in the parent optimizer class and are
    # global (Cannot be set per group).
    _child_vars = ["max_weight_norm", "loss_scaling"]
    # All the attributes and variables which don't exist in the parent optimizer class.
    _child_only = _child_vars + [
        "accum_type", "first_order_momentum_accum_type",
        "second_order_momentum_accum_type"
    ]
    # Attributes (from the parent or child class) which can be set per group.
    _group_vars = ["lr", "weight_decay", "betas", "eps"]

    def __init__(self,
                 params,
                 lr=None,
                 betas=None,
                 eps=None,
                 weight_decay=None,
                 bias_correction=None,
                 loss_scaling=None,
                 max_weight_norm=None,
                 accum_type=None,
                 first_order_momentum_accum_type=None,
                 second_order_momentum_accum_type=None):
        """
        :param iterable params: parameters to optimize.
        :param lr: learning rate
        :type lr: float, optional
        :param betas: (beta1, beta2) parameters used in LAMB.
        :type betas: tuple, optional
        :param eps: term added to the denominator to ensure numerical
           stability/
        :type eps: float, optional
        :param weight_decay: weight decay factor.
        :type weight_decay: float, optional
        :param bias_correction: True: compute LAMB with bias correction.
        :type bias_correction: bool, optional
        :param loss_scaling: Factor by which to scale the loss and hence
            gradients to assist numerical stability when using float16.
        :type loss_scaling: float, optional
        :param max_weight_norm: maximum value of the output of scaling
            function, phi(). Set to None to disable scaling function.
        :type max_weight_norm: float, optional
        :param accum_type: data type used for gradients.
        :type accum_type: torch.dtype, optional
        :param first_order_momentum_accum_type: data type used to store
            the first order momentum values for each parameter.
        :type first_order_momentum_accum_type: torch.dtype, optional
        :param second_order_momentum_accum_type: data type used to store
           the second order momentum values for each parameter.
        :type second_order_momentum_accum_type: torch.dtype, optional
        """
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
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

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

    def step(self, closure=None):
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

                if group["bias_correction"]:
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

                if r1 == 0 or r2 == 0:
                    trust = 1.0
                else:
                    trust = r1.clamp(max=self.max_weight_norm) / r2

                p.data.add_(upd, alpha=-group['lr'] * trust)

        return loss

    def __getstate__(self):
        state = super().__getstate__()
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
        return state


def _check_constructor_match_parent(child_class):
    parent = child_class.__bases__[0]
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
