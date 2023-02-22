# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import torch

from poptorch_geometric import TrainingStepper, set_aggregation_dim_size


def conv_harness(conv,
                 dataset=None,
                 edge_index_max=None,
                 post_proc=None,
                 loss_fn=torch.nn.MSELoss(),
                 num_steps=4,
                 atol=1e-5,
                 rtol=1e-4,
                 batch=None):
    class ConvWrapper(torch.nn.Module):
        def __init__(self, conv, loss_fn, post_proc=None):
            super().__init__()
            self.conv = conv
            self.loss_fn = loss_fn
            self.post_proc = post_proc

        def forward(self, *args):
            x = self.conv(*args)
            if self.post_proc is not None:
                x = self.post_proc(x)
            if self.training:
                target = torch.ones_like(x)
                loss = self.loss_fn(x, target)
                return x, loss

            return x

    model = ConvWrapper(conv, loss_fn=loss_fn, post_proc=post_proc)

    if batch is None and dataset is not None:
        batch = (dataset.x, dataset.edge_index)

    if edge_index_max is None:
        set_aggregation_dim_size(model, int(dataset.edge_index.max()) + 1)

    stepper = TrainingStepper(model, atol=atol, rtol=rtol)

    stepper.run(num_steps, batch)
