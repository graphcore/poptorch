#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch
import torch.optim as optim


def test_inferenceBatching():
    torch.manual_seed(42)

    model = torch.nn.Linear(6, 20)

    # Actually batched by 100.
    input = torch.randn([10, 1, 5, 6])

    # Run pytorch native on CPU batchsize 10.
    nativeOutput = model(input)

    # Run on IPU batch size 1 * 10 popart batches.
    ipuModel = poptorch.inferenceModel(model, device_iterations=10)
    poptorchOut = ipuModel(input)

    assert torch.allclose(poptorchOut, nativeOutput, atol=1e-07)


def test_trainingBatching():
    torch.manual_seed(4424242)

    # 10 Batches of 10.
    input = torch.randn(10, 10)

    # 10 batches of 1
    label = torch.randint(0, 10, [1])
    label = label.expand([10])
    model = torch.nn.Linear(10, 10)

    # Run on IPU batch size 1 * 10 popart batches.
    poptorch_model = poptorch.trainingModel(model,
                                            device_iterations=10,
                                            loss=torch.nn.CrossEntropyLoss())

    # Run all 10 batches as batchsize 10.
    out = model(input)

    # Sanity check we weren't already matching the label.
    assert not torch.equal(torch.argmax(out, dim=1), label)

    for i in range(0, 1000):
        poptorchOut, loss = poptorch_model(input, label)

        # Each batch should report its own loss.
        assert len(loss.size()) == 1
        assert loss.size()[0] == 10

    # Run with trained weights.
    out = model(input)

    # Check we are now equal with labels.
    assert torch.equal(torch.argmax(out, dim=1), label)
