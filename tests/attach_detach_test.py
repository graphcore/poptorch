#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import math
import pytest
import torch
import helpers
import poptorch


def test_attach_detach():
    torch.manual_seed(42)

    target = torch.randint(0, 10, [1])
    target = target.expand([10])
    input = torch.randn(10, 10)

    model = torch.nn.Linear(10, 10)

    opts = poptorch.Options()
    # Ensure that both models use the same IPU
    opts.useIpuId(1)

    training = helpers.trainingModelWithLoss(model,
                                             options=opts,
                                             loss=torch.nn.CrossEntropyLoss())

    inference = poptorch.inferenceModel(model, options=opts)

    _, initial_loss = training(input, target)

    if math.isnan(initial_loss):
        raise ValueError("original_loss is NaN")

    if poptorch.ipuHardwareIsAvailable():
        with pytest.raises(RuntimeError) as excinfo:
            inference.compile(torch.randn(10))
            assert excinfo.match("Failed to acquire")

    training.detachFromDevice()
    # Ensure that this breaks
    with pytest.raises(AssertionError):
        training.detachFromDevice()

    inference.compile(torch.randn(10))

    if poptorch.ipuHardwareIsAvailable():
        inference.detachFromDevice()

    assert initial_loss > 0.1

    loss = float('nan')

    for _ in range(0, 2):
        _, loss = training(input, target)
        # Each batch should NOT report its own loss. As by default training
        # model should have a "Final" anchor.
        assert len(loss.size()) == 0

        if math.isnan(loss):
            raise ValueError("loss is NaN")

        training.detachFromDevice()

        inference(torch.randn(10))
        inference.detachFromDevice()
