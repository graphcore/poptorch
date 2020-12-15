# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import math
import inspect
import torch
from torch.optim.optimizer import required


class SGD(torch.optim.SGD):
    """ Stochastic gradient descent with optional momentum.

    The optimizer matches PyTorch's implementation (torch.optim.SGD) with
    optional loss and velocity scaling.

    Nesterov momentum is not currently supported.
    """

    def __init__(self,
                 params,
                 lr=required,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False,
                 loss_scaling=1.0,
                 velocity_scaling=1.0):
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
        super().__init__(params, lr, momentum, dampening, weight_decay,
                         nesterov)

        self.defaults["loss_scaling"] = loss_scaling
        self.defaults["velocity_scaling"] = velocity_scaling
        for group in self.param_groups:
            group.setdefault("loss_scaling", loss_scaling)
            group.setdefault("velocity_scaling", velocity_scaling)


class Adam(torch.optim.Adam):
    """ Adam optimizer.

    This optimizer matches PyTorch's implementation (torch.optim.Adam) with
    optional loss scaling.

    AMSGrad is currently not supported."""

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 loss_scaling=1.0,
                 accumType=torch.float32,
                 firstOrderMomentumAccumType=torch.float32,
                 secondOrderMomentumAccumType=torch.float32):
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
        :param accumType: data type used for gradients.
        :type accumType: torch.dtype, optional
        :param firstOrderMomentumAccumType: data type used to store
            the first order momentum values for each parameter.
        :type firstOrderMomentumAccumType: torch.dtype, optional
        :param secondOrderMomentumAccumType: data type used to store
            the second order momentum values for each parameter.
        :type secondOrderMomentumAccumType: torch.dtype, optional
        """
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)

        self.defaults["loss_scaling"] = loss_scaling

        supportedTypes = [torch.float16, torch.float32]
        errString = ("Accumulation types must be either torch.float32"
                     " or torch.float16")
        assert accumType in supportedTypes, errString
        assert firstOrderMomentumAccumType in supportedTypes, errString
        assert secondOrderMomentumAccumType in supportedTypes, errString

        self.accumType = accumType == torch.float16
        self.firstOrderMomentumAccumType = \
             firstOrderMomentumAccumType == torch.float16
        self.secondOrderMomentumAccumType = \
             secondOrderMomentumAccumType == torch.float16

        for group in self.param_groups:
            group.setdefault("loss_scaling", loss_scaling)


class AdamW(torch.optim.AdamW):
    """ Adam optimizer with true weight decay.

    This optimizer matches PyTorch's implementation (torch.optim.AdamW) with
    optional loss scaling.

    AMSGrad is currently not supported."""

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0.01,
                 amsgrad=False,
                 loss_scaling=1.0,
                 biasCorrection=True,
                 accumType=torch.float32,
                 firstOrderMomentumAccumType=torch.float32,
                 secondOrderMomentumAccumType=torch.float32):
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
        :param accumType: data type used for gradients.
        :type accumType: torch.dtype, optional
        :param firstOrderMomentumAccumType: data type used to store
            the first order momentum values for each parameter.
        :type firstOrderMomentumAccumType: torch.dtype, optional
        :param secondOrderMomentumAccumType: data type used to store
            the second order momentum values for each parameter.
        :type secondOrderMomentumAccumType: torch.dtype, optional
        """
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)

        self.defaults["loss_scaling"] = loss_scaling
        self.defaults["biasCorrection"] = biasCorrection

        supportedTypes = [torch.float16, torch.float32]
        errString = ("Accumulation types must be either torch.float32"
                     " or torch.float16")
        assert accumType in supportedTypes, errString
        assert firstOrderMomentumAccumType in supportedTypes, errString
        assert secondOrderMomentumAccumType in supportedTypes, errString

        self.accumType = accumType == torch.float16
        self.firstOrderMomentumAccumType = \
             firstOrderMomentumAccumType == torch.float16
        self.secondOrderMomentumAccumType = \
             secondOrderMomentumAccumType == torch.float16

        for group in self.param_groups:
            group.setdefault("loss_scaling", loss_scaling)
            group.setdefault("biasCorrection", biasCorrection)


class RMSprop(torch.optim.RMSprop):
    """ RMSprop optimizer with optional L2 penalty.

    This optimizer matches PyTorch's implementation (torch.optim.RMSprop) with
    optional loss scaling."""

    def __init__(self,
                 params,
                 lr=1e-2,
                 alpha=0.99,
                 eps=1e-8,
                 weight_decay=0,
                 momentum=0,
                 centered=False,
                 loss_scaling=1.0):
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
        """
        super().__init__(params, lr, alpha, eps, weight_decay, momentum,
                         centered)

        self.defaults["loss_scaling"] = loss_scaling
        for group in self.param_groups:
            group.setdefault("loss_scaling", loss_scaling)


class LAMB(torch.optim.Optimizer):
    """ Layer-wise Adaptive Moments (LAMB) optimizer (biased version).

        Based on "Large Batch Optimization for Deep Learning: Training BERT
        in 76 minutes" (https://arxiv.org/abs/1904.00962).

        The scaling function phi(z) is fixed as min(z, max_weight_norm);
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=1e-2,
                 biasCorrection=True,
                 loss_scaling=1.0,
                 max_weight_norm=None,
                 accumType=torch.float32,
                 firstOrderMomentumAccumType=torch.float32,
                 secondOrderMomentumAccumType=torch.float32):
        """
        :param iterable params: parameters to optimize.
        :param lr: learning rate
        :type lr: float, optional
        :param betas: (beta1, beta2) parameters used in LAMB.
        :type betas: tuple, optional
        :param eps: term added to the denominator to ensure numerical
           stability/
        :type eps: float, optional
        :param weight_decay: (AdamW) weight decay factor.
        :type weight_decay: float, optional
        :param biasCorrection: True: compute LAMB with bias correction.
        :type biasCorrection: bool, optional
        :param loss_scaling: Factor by which to scale the loss and hence
            gradients to assist numerical stability when using float16.
        :type loss_scaling: float, optional
        :param max_weight_norm: maximum value of the output of scaling
            function, phi(). Set to None to disable scaling function.
        :type max_weight_norm: float, optional
        :param accumType: data type used for gradients.
        :type accumType: torch.dtype, optional
        :param firstOrderMomentumAccumType: data type used to store
            the first order momentum values for each parameter.
        :type firstOrderMomentumAccumType: torch.dtype, optional
        :param secondOrderMomentumAccumType: data type used to store
           the second order momentum values for each parameter.
        :type secondOrderMomentumAccumType: torch.dtype, optional
        """
        if max_weight_norm is None:
            max_weight_norm = 65500.0  # FP16 Max
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        biasCorrection=biasCorrection,
                        loss_scaling=loss_scaling,
                        max_weight_norm=max_weight_norm)
        super().__init__(params, defaults)

        supportedTypes = [torch.float16, torch.float32]
        errString = """Accumulation types must be either torch.float32
                or torch.float16"""
        assert accumType in supportedTypes, errString
        assert firstOrderMomentumAccumType in supportedTypes, errString
        assert secondOrderMomentumAccumType in supportedTypes, errString

        self.accumType = accumType == torch.float16
        self.firstOrderMomentumAccumType = \
             firstOrderMomentumAccumType == torch.float16
        self.secondOrderMomentumAccumType = \
             secondOrderMomentumAccumType == torch.float16

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("biasCorrection", self.defaults["biasCorrection"])
            group.setdefault("loss_scaling", self.defaults["loss_scaling"])

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

                if group["biasCorrection"]:
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
                    trust = r1.clamp(max=group["max_weight_norm"]) / r2

                p.data.add_(upd, alpha=-group['lr'] * trust)

        return loss


def _check_constructor_match_parent(child_class, extra_args=None):
    parent = child_class.__bases__[0]
    parent_params = inspect.signature(parent.__init__).parameters
    child_params = inspect.signature(child_class.__init__).parameters
    extra_args = extra_args or []
    assert len(parent_params) + len(extra_args) == len(child_params), (
        f"Expected {len(parent_params) + len(extra_args)} parameters but got "
        f"{len(child_params)}")

    child_params = iter(child_params.items())
    for idx, (_, param) in enumerate(parent_params.items()):
        _, child_param = next(child_params)
        assert child_param == param, (f"Mismatch for parameter {idx}: expected"
                                      f"'{param}' but got '{child_param}'")

    for extra_arg in extra_args:
        name, _ = next(child_params)
        assert name == extra_arg, (f"Expected an extra argument named "
                                   f"'{extra_arg}' but got '{name}'")


_check_constructor_match_parent(SGD, ["loss_scaling", "velocity_scaling"])
_check_constructor_match_parent(Adam, [
    "loss_scaling", "accumType", "firstOrderMomentumAccumType",
    "secondOrderMomentumAccumType"
])
_check_constructor_match_parent(AdamW, [
    "loss_scaling", "biasCorrection", "accumType",
    "firstOrderMomentumAccumType", "secondOrderMomentumAccumType"
])
_check_constructor_match_parent(RMSprop, ["loss_scaling"])
