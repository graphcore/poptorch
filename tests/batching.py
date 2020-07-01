#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch
import torch.optim as optim

import pytest


def test_inferenceBatching():
    torch.manual_seed(42)

    model = torch.nn.Linear(6, 20)

    # Actually batched by 100.
    input = torch.randn([10, 1, 5, 6])

    # Run pytorch native on CPU batchsize 10.
    nativeOutput = model(input)

    # Run on IPU batch size 1 * 10 popart batches.
    opts = poptorch.Options().deviceIterations(10)
    ipuModel = poptorch.inferenceModel(model, opts)
    poptorchOut = ipuModel(input)

    # Check that inference wrapper has defaulted to "All".
    assert len(poptorchOut.size()) == 4
    assert poptorchOut.size()[0] == 10
    torch.testing.assert_allclose(poptorchOut, nativeOutput)


def test_trainingBatching():
    torch.manual_seed(4424242)

    # 10 Batches of 10.
    input = torch.randn(10, 10)

    # 10 batches of 1
    label = torch.randint(0, 10, [1])
    label = label.expand([10])
    model = torch.nn.Linear(10, 10)

    # Run on IPU batch size 1 * 10 popart batches.
    opts = poptorch.Options().deviceIterations(10)
    poptorch_model = poptorch.trainingModel(model,
                                            opts,
                                            loss=torch.nn.CrossEntropyLoss())

    # Run all 10 batches as batchsize 10.
    out = model(input)

    # Sanity check we weren't already matching the label.
    assert not torch.equal(torch.argmax(out, dim=1), label)

    for i in range(0, 1000):
        poptorchOut, loss = poptorch_model(input, label)

        # Each batch should NOT report its own loss. As by default training model should have a "Final" anchor.
        assert len(loss.size()) == 1
        assert loss.size()[0] == 1

    # Copy weights back to host model.
    poptorch_model.copyWeightsToHost()

    # Run with trained weights.
    out = model(input)

    # Check we are now equal with labels.
    assert torch.equal(torch.argmax(out, dim=1), label)


@pytest.mark.parametrize("anchor", list(poptorch.AnchorMode))
def test_inferenceAnchors(anchor):
    torch.manual_seed(42)

    model = torch.nn.Linear(6, 20)

    # Actually batched by 100.
    input = torch.randn([10, 1, 5, 6])

    # Run pytorch native on CPU batchsize 10.
    nativeOutput = model(input)

    # Run on IPU batch size 1 * 10 popart batches. anchor_return_period ignored if not EVERYN
    opts = poptorch.Options().deviceIterations(10)
    opts.anchorMode(anchor, anchor_return_period=5)
    ipuModel = poptorch.inferenceModel(model, opts)
    poptorchOut = ipuModel(input)

    if anchor in [poptorch.AnchorMode.All, poptorch.AnchorMode.Default]:
        # Expect the full batch.
        assert len(poptorchOut.size()) == 4
        assert poptorchOut.size()[0] == 10
        torch.testing.assert_allclose(poptorchOut, nativeOutput)
    elif anchor == poptorch.AnchorMode.EveryN:
        # Otherwise we are expecting device_iterations / N
        assert len(poptorchOut.size()) == 4
        assert poptorchOut.size()[0] == 2

        # Check each N is the correct batch
        torch.testing.assert_allclose(poptorchOut[0], nativeOutput[4])
        torch.testing.assert_allclose(poptorchOut[1], nativeOutput[9])

    else:
        # Otherwise we are expecting just one element per batch.
        assert len(poptorchOut.size()) == 4
        assert poptorchOut.size()[0] == 1

        if anchor == poptorch.AnchorMode.Final:
            # Check we are the same as the last output.
            torch.testing.assert_allclose(poptorchOut, nativeOutput[-1])
        elif anchor == poptorch.AnchorMode.Sum:
            # Check we are close to the sum of the batch dim.
            sum = torch.sum(nativeOutput, dim=0, keepdim=True)
            torch.testing.assert_allclose(poptorchOut, sum)
        else:
            assert False, "Unexpected anchor type %s" % anchor


@pytest.mark.parametrize("anchor", list(poptorch.AnchorMode))
def test_trainingAnchors(anchor):
    torch.manual_seed(42)

    # 1000 Batches of 10.
    input = torch.randn(1000, 10)

    # 1000 batches of 1
    label = torch.randint(0, 10, [1])
    label = label.expand([1000])

    # The model
    model = torch.nn.Linear(10, 10)

    # Run pytorch native on CPU batchsize 10.
    nativeOutput = model(input)

    # Run on IPU batch size 1 * 1000 popart batches.
    opts = poptorch.Options().deviceIterations(1000)
    opts.anchorMode(anchor, anchor_return_period=20)
    poptorch_model = poptorch.trainingModel(model,
                                            opts,
                                            loss=torch.nn.CrossEntropyLoss())

    poptorchOut, loss = poptorch_model(input, label)

    if anchor == poptorch.AnchorMode.All:
        # Expect the full batch.
        assert len(poptorchOut.size()) == 2
        assert poptorchOut.size()[0] == 1000

        assert len(loss.size()) == 1
        assert loss.size()[0] == 1000

        # Check the rolling average loss is downward sloped.
        interval = 100
        previous_average = torch.mean(loss[:interval])
        for i in range(1, 1000 // interval):
            start = interval * i
            end = start + interval
            new_average = torch.mean(loss[start:end])

            assert new_average < previous_average

            previous_average = new_average

    elif anchor == poptorch.AnchorMode.EveryN:
        # Otherwise we are expecting device_iterations / N
        assert len(poptorchOut.size()) == 2
        assert poptorchOut.size()[0] == 50

        # There's too much noise in the losses for us to test directly without averaging like above so just test sizes.
        assert len(loss.size()) == 1
        assert loss.size()[0] == 50
    else:
        # Otherwise we are expecting just one element per batch.
        assert len(poptorchOut.size()) == 2
        assert poptorchOut.size()[0] == 1

        assert len(loss.size()) == 1
        assert loss.size()[0] == 1

        if anchor in [poptorch.AnchorMode.Final, poptorch.AnchorMode.Default]:
            # We just have to check the loss is small.
            # This is just relative to the previously observed loss values on this test with this seed.
            assert loss < 0.2

        elif anchor == poptorch.AnchorMode.Sum:
            # We just have to check that the loss is huge.
            assert loss > 500.0
        else:
            assert False, "Unexpected anchor type %s" % anchor
