# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import List
import torch

from poptorch_geometric import TrainingStepper


def aggr_harness(aggr,
                 dim_size,
                 dataloader=None,
                 post_proc=None,
                 sorted_index=False,
                 loss_fn=torch.nn.MSELoss(),
                 num_steps=4,
                 atol=1e-5,
                 rtol=1e-4):
    class AggrWrapper(torch.nn.Module):
        def __init__(self, aggr, loss_fn, post_proc=None):
            super().__init__()
            self.aggr = aggr
            self.loss_fn = loss_fn
            self.post_proc = post_proc

        def forward(self, *args):
            x = args[0]
            edge_index = args[1]
            size = args[2]

            broadcast_index = edge_index[1] if sorted_index else edge_index[0]
            aggr_index = edge_index[0] if sorted_index else edge_index[1]

            x_broadcasted = torch.index_select(x, 0, broadcast_index)
            result = self.aggr(x_broadcasted, aggr_index, dim_size=size)

            if self.post_proc is not None:
                if isinstance(result, List):
                    result = torch.cat(result)
                result = self.post_proc(result)

            if self.training:
                if isinstance(result, List):
                    result = torch.cat(result)
                target = torch.ones_like(result)
                loss = self.loss_fn(result, target)
                return result, loss

            return result

    model = AggrWrapper(aggr, loss_fn=loss_fn, post_proc=post_proc)

    stepper = TrainingStepper(model, atol=atol, rtol=rtol)

    if dataloader is not None:
        for step, batch in enumerate(dataloader):
            if step == num_steps:
                break
            stepper.run(1, (batch.x, batch.edge_index, dim_size))
