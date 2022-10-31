#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import copy
import torch
import pytest
import helpers

from poptorch.experimental import ipu_wrapper

all_torch_optimisers = []
for candidate_str in dir(torch.optim):
    candidate = getattr(torch.optim, candidate_str)
    if not isinstance(candidate, type) or candidate == torch.optim.Optimizer:
        continue
    if not issubclass(candidate, torch.optim.Optimizer):
        continue
    all_torch_optimisers.append(candidate)


class OptimizerTestModel(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.layers = [torch.nn.Linear(num_channels, num_channels)]
        self.model = torch.nn.Sequential(*self.layers)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, data, target):
        out = self.model(data)
        loss = self.loss(out, target)
        return out, loss


@pytest.mark.parametrize("optim", all_torch_optimisers)
def test_optimizer(optim):
    if optim not in (torch.optim.SGD, torch.optim.ASGD):
        # TODO T62506
        pytest.skip()

    N = 5
    C = 10

    model = OptimizerTestModel(C)

    torch.manual_seed(42)
    for layer in model.model:
        if not hasattr(layer, "weight"):
            continue
        torch.nn.init.uniform_(layer.weight, 0.1, 2.0)
        torch.nn.init.uniform_(layer.bias, -1.0, 1.0)

    def training_step(model, optim, t, label):
        optim.zero_grad()
        _, loss = model(t, label)
        loss.backward()
        optim.step()

        return loss

    ipu_training_step = ipu_wrapper(training_step)

    cpu_model = copy.deepcopy(model)

    model.to("xla")

    # Embed optimiser in a list to allow construction within first training step
    cpu_optim = optim(cpu_model.parameters(), lr=0.1)
    ipu_optim = optim(model.parameters(), lr=0.1)

    t = torch.rand([N, C])
    label = torch.randint(0, C, [N])

    training_steps = 10
    for _ in range(training_steps):
        t = torch.rand([N, C])
        label = torch.randint(0, C, [N])

        training_step(cpu_model, cpu_optim, t, label)
        ipu_training_step(model, ipu_optim, t.to("xla"), label.int().to("xla"))

        ipu_params_cpu = [param.cpu() for param in model.parameters()]

        for cpu, ipu in zip(cpu_model.parameters(), ipu_params_cpu):
            helpers.assert_allclose(expected=cpu, actual=ipu, equal_nan=True)
