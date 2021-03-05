#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os  # pylint: disable=unused-import
import unittest.mock
import torch
import torchvision.models as models
import helpers
import poptorch


def test_half_float_default_option():
    class SimpleAdder(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

    t1 = torch.tensor([1.]).half()
    t2 = torch.tensor([2.]).float()

    outHalf = inference_model(t1, t2)
    assert outHalf.dtype == torch.half

    # Refresh and try the other way
    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

    outHalf = inference_model(t2, t1)
    assert outHalf.dtype == torch.half


def test_half_float_downcast_option():
    class SimpleAdder(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = SimpleAdder()
    opts = poptorch.Options()
    opts.Precision.halfFloatCasting(
        poptorch.HalfFloatCastingBehavior.FloatDowncastToHalf)
    inference_model = poptorch.inferenceModel(model, opts)

    t1 = torch.tensor([1.]).half()
    t2 = torch.tensor([2.]).float()

    outHalf = inference_model(t1, t2)
    assert outHalf.dtype == torch.half

    # Refresh and try the other way
    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

    outHalf = inference_model(t2, t1)
    assert outHalf.dtype == torch.half


def test_half_float_upcast_option():
    class SimpleAdder(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = SimpleAdder()
    opts = poptorch.Options()
    opts.Precision.halfFloatCasting(
        poptorch.HalfFloatCastingBehavior.HalfUpcastToFloat)
    inference_model = poptorch.inferenceModel(model, opts)

    t1 = torch.tensor([1.]).half()
    t2 = torch.tensor([2.]).float()

    outFlot = inference_model(t1, t2)
    assert outFlot.dtype == torch.float

    # Refresh and try the other way
    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model, opts)

    outFloat = inference_model(t2, t1)
    assert outFloat.dtype == torch.float


@unittest.mock.patch.dict("os.environ", helpers.disableSmallModel())
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

    helpers.assert_allclose(expected=model(t1),
                            actual=out.float(),
                            rtol=0.001,
                            atol=1e-04)


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
    hidden = (
        torch.randn(1, 1, numHidden).half(),
        torch.randn(1, 1, numHidden).half(),
    )
    ipuOut = ipuLstm(inputs, hidden)
    assert isinstance(ipuOut[0], torch.HalfTensor)


def test_ipu_print_tensor():
    class SimplePrinter(torch.nn.Module):
        def forward(self, x):
            return poptorch.ipu_print_tensor(x)

    t1 = torch.tensor([1.], dtype=torch.float16)
    inference_model = poptorch.inferenceModel(SimplePrinter())
    out = inference_model(t1)
    assert out == 1.0
    assert out.dtype == torch.float16


# pylint: disable=protected-access
def test_half_tracing():
    def check_param_types(module, dtype):
        for param in module.parameters():
            assert param.dtype == dtype

    torch.manual_seed(42)

    x = torch.randn(10, 10)
    model = torch.nn.Sequential()
    model.add_module('linear1', torch.nn.Linear(10, 10))
    model.add_module('linear2', torch.nn.Linear(10, 10))

    popmodel = poptorch.inferenceModel(model)
    popmodel(x)
    check_param_types(popmodel._trace.linear1, torch.float)
    check_param_types(popmodel._trace.linear2, torch.float)

    model.linear2.half()
    popmodel = poptorch.inferenceModel(model)
    popmodel(x)
    check_param_types(popmodel._trace.linear1, torch.float)
    check_param_types(popmodel._trace.linear2, torch.half)

    model.half()
    popmodel = poptorch.inferenceModel(model)
    popmodel(x)
    check_param_types(popmodel._trace.linear1, torch.half)
    check_param_types(popmodel._trace.linear2, torch.half)

    model.linear1.float()
    popmodel = poptorch.inferenceModel(model)
    popmodel(x)
    check_param_types(popmodel._trace.linear1, torch.float)
    check_param_types(popmodel._trace.linear2, torch.half)

    model.half()
    model.linear2.float()
    check_param_types(popmodel._trace.linear1, torch.half)
    check_param_types(popmodel._trace.linear2, torch.float)
