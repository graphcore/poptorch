# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import List
import torch

from poptorch_geometric import TrainingStepper


def kge_harness(kge,
                dataloader,
                post_proc=None,
                loss_fn=torch.nn.MSELoss(),
                num_steps=4,
                atol=5e-3,
                rtol=5e-3,
                equal_nan=False,
                enable_fp_exception=True):
    class KgeWrapper(torch.nn.Module):
        def __init__(self, kge, loss_fn, post_proc=None):
            super().__init__()
            self.model = kge
            self.loss_fn = loss_fn
            self.post_proc = post_proc

        def forward(self, *args):
            result = self.model(*args)

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

    model = KgeWrapper(kge, loss_fn=loss_fn, post_proc=post_proc)

    stepper = TrainingStepper(model,
                              atol=atol,
                              rtol=rtol,
                              equal_nan=equal_nan,
                              enable_fp_exception=enable_fp_exception)

    if dataloader is not None:
        for step, batch in enumerate(dataloader):
            if step == num_steps:
                break
            stepper.run(1, batch)
