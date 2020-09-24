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
                 velocity_scaling=1.0):
        super().__init__(params, lr, momentum, dampening, weight_decay,
                         nesterov)
        self.velocity_scaling = velocity_scaling


_impl.check_constructor_match_parent(SGD, ["velocity_scaling"])
