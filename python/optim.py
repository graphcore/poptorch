# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
from torch.optim.optimizer import required

from . import _impl


class SGD(torch.optim.SGD):
    def __init__(self,
                 params,
                 lr=required,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False,
                 loss_scaling=1.0,
                 velocity_scaling=1.0):
        super().__init__(params, lr, momentum, dampening, weight_decay,
                         nesterov)
        self.loss_scaling = loss_scaling
        self.velocity_scaling = velocity_scaling


class Adam(torch.optim.Adam):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 loss_scaling=1.0):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.loss_scaling = loss_scaling


_impl.check_constructor_match_parent(SGD, ["loss_scaling", "velocity_scaling"])
_impl.check_constructor_match_parent(Adam, ["loss_scaling"])
