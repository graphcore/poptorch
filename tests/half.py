#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import poptorch
import pytest
import torch
import torchvision.models as models
import helpers


def test_resnet():
    torch.manual_seed(42)

    image_input = torch.randn([1, 3, 224, 224]).half()
    t1 = torch.tensor([1.]).long()
    # We are running on a dummy input so it doesn't matter if the weights are trained.
    model = models.resnet18(pretrained=False)
    model.train()
    model.half()

    training_model = helpers.trainingModelWithLoss(model,
                                                   loss=torch.nn.NLLLoss())

    # Run on IPU.
    poptorch_out, loss = training_model(image_input, t1)

    assert poptorch_out.dtype == torch.half
    assert loss.dtype == torch.half


def test_model_with_weights():
    model = torch.nn.Linear(1, 10).half()
    t1 = torch.tensor([1.]).half()

    inference_model = poptorch.inferenceModel(model)
    out = inference_model(t1)

    assert out.dtype == torch.half

    # For running on host.
    model = model.float()
    t1 = t1.float()

    torch.testing.assert_allclose(model(t1),
                                  out.float(),
                                  rtol=0.001,
                                  atol=1e-04)


@pytest.mark.skip(reason="TODO(T21014)")
def test_simple_model():
    class SimpleAdder(torch.nn.Module):
        def forward(self, x, y, z, w):
            return x + y + 5, z + w + 5

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

    t1 = torch.tensor([1.]).half()
    t2 = torch.tensor([2.]).half()

    t3 = torch.tensor([3.])
    t4 = torch.tensor([4.])

    outHalf, outFloat = inference_model(t1, t2, t3, t4)

    assert outHalf.dtype == torch.half
    assert outHalf.float() == 8.0

    assert outFloat.dtype == torch.float
    assert outFloat == 12.0


def test_lstm():
    torch.manual_seed(42)
    numHidden = 5
    inputSize = 3
    lstm = torch.nn.LSTM(3, numHidden)
    lstm.half()
    ipuLstm = poptorch.inferenceModel(lstm)
    inputs = [torch.randn(1, inputSize).half() for _ in range(5)]
    # Add the extra 2nd dimension
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1,
                          numHidden).half(), torch.randn(1, 1,
                                                         numHidden).half())
    ipuOut = ipuLstm(inputs, hidden)
    assert isinstance(ipuOut[0], torch.HalfTensor)
