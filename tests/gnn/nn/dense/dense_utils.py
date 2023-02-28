# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch

from poptorch_geometric import TrainingStepper


def dense_harness(dense,
                  batch=None,
                  post_proc=None,
                  loss_fn=torch.nn.MSELoss(),
                  num_steps=4,
                  atol=1e-5,
                  rtol=1e-4):
    class DenseWrapper(torch.nn.Module):
        def __init__(self, dense, loss_fn, post_proc=None):
            super().__init__()
            self.dense = dense
            self.loss_fn = loss_fn
            self.post_proc = post_proc

        def forward(self, *args):
            x = self.dense(*args)
            if self.post_proc is not None:
                x = self.post_proc(x)
            if self.training:
                target = torch.ones_like(x)
                loss = self.loss_fn(x, target)
                return x, loss

            return x

    model = DenseWrapper(dense, loss_fn=loss_fn, post_proc=post_proc)

    stepper = TrainingStepper(model, atol=atol, rtol=rtol)

    stepper.run(num_steps, batch)
