#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch


# Test custom loss by training to a target
def test_custom_loss():
    torch.manual_seed(42)

    model = torch.nn.Linear(10, 10)

    class CustomLoss(torch.nn.Module):
        # Mean squared error scaled.
        def forward(self, x, target):
            loss = x - target
            loss = loss * loss * 5.0
            return poptorch.identity_loss(loss, reduction="mean")

    loss = CustomLoss()

    poptorch_model = poptorch.trainingModel(model,
                                            loss=loss)

    target = torch.randn(10)
    input = torch.randn(10)

    # Make sure the first run doesn't already pass the test.
    original, original_loss = poptorch_model(input, target)

    assert original_loss > 0.1
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    # Pytorch native.
    out = model(input)
    assert not torch.allclose(out, target, rtol=1e-02, atol=1e-02)

    for i in range(0, 2500):
        out, loss = poptorch_model(input, target)

    # Check we have trained the "model"
    assert loss < 0.001
    assert torch.allclose(out, target, rtol=1e-02, atol=1e-02)

    # Copy weights back to host.
    poptorch_model.copyWeightsToHost()

    # Check that the pytorch native model is also returning the trained
    # value that was trained on IPU.
    out = model(input)
    assert torch.allclose(out, target, rtol=1e-02, atol=1e-02)
