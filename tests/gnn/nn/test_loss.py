# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch.nn import L1Loss, MSELoss

from poptorch_geometric import TrainingStepper


def loss_harness(in_channels,
                 out_channels,
                 cpu_dataloader=None,
                 ipu_dataloader=None,
                 loss_fn=None,
                 num_steps=4,
                 atol=5e-4,
                 rtol=5e-4):
    class LinearModel(torch.nn.Module):
        def __init__(self, loss_fn):
            assert loss_fn is not None
            assert hasattr(loss_fn, 'reduction')
            super().__init__()
            self.loss = loss_fn
            self.linear = torch.nn.Linear(in_channels, out_channels)

        def forward(self, *args):
            x = args[0]
            nodes_mask = args[1]
            target = args[2]

            result = self.linear(x)
            # Apply nodes mask, so that the loss may be computed properly
            if nodes_mask is not None:
                result[~nodes_mask] = 0

            if self.training:
                # target = torch.ones_like(result)
                if nodes_mask is not None:
                    target[~nodes_mask] = 0

                loss = self.loss(result, target)
                # In case, the loss function applies mean reduction, the result
                # has to be rescaled by the effective size of the batch
                # (excluding padding).
                if nodes_mask is not None and self.loss.reduction == 'mean':
                    size = nodes_mask.shape[0]
                    real_size = torch.count_nonzero(nodes_mask)
                    loss = loss * size / real_size

                return (result, loss)
            return result

    model = LinearModel(loss_fn)
    stepper = TrainingStepper(model, atol=atol, rtol=rtol)

    if cpu_dataloader is not None and ipu_dataloader is not None:
        for step, (cpu_batch,
                   ipu_batch) in enumerate(zip(cpu_dataloader,
                                               ipu_dataloader)):
            if step == num_steps:
                break
            stepper.run(1, (cpu_batch.x, None,
                            torch.ones(cpu_batch.x.shape[0], out_channels)),
                        (ipu_batch.x, ipu_batch.nodes_mask,
                         torch.ones(ipu_batch.x.shape[0], out_channels)))


@pytest.mark.parametrize('loss_fn', [
    L1Loss,
    MSELoss,
])
@pytest.mark.parametrize('reduction', ['mean', 'sum'])
def test_loss_fixedsize_vs_regular_dataloader(loss_fn, reduction, dataloader,
                                              fixed_size_dataloader):
    first_sample = next(iter(dataloader))
    in_channels = first_sample.num_node_features
    out_channels = in_channels * 2

    loss_harness(in_channels,
                 out_channels,
                 cpu_dataloader=dataloader,
                 ipu_dataloader=fixed_size_dataloader,
                 loss_fn=loss_fn(reduction=reduction))
