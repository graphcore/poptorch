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
    assert outHalf.dtype == torch.float

    # Refresh and try the other way
    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

    outHalf = inference_model(t2, t1)
    assert outHalf.dtype == torch.float


@unittest.mock.patch.dict("os.environ", helpers.disableSmallModel())
def test_resnet():
    torch.manual_seed(42)

    image_input = torch.randn([1, 3, 224, 224]).half()
    t1 = torch.tensor([1.]).long()
    loss_fn = torch.nn.NLLLoss()

    class ModelWithLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # We are running on a dummy input so it doesn't matter whether the
            # weights are trained.
            self.base_model = models.resnet18(pretrained=False)

        def forward(self, data, target):
            out = self.base_model(data)
            loss = loss_fn(out, target)
            return out, loss

    model = ModelWithLoss()
    model.train()
    model.half()

    training_model = poptorch.trainingModel(model)

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


def test_buffers():
    torch.manual_seed(42)
    fake_data = torch.ones(1, 64, 10, 10).half()

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(64)

            self.bn.running_mean += torch.randn(64)
            self.bn.running_var += torch.randn(64)

        def forward(self, i):
            out = self.bn(i)
            return out, self.bn.running_var, self.bn.running_mean

    model = M()

    cpu_mean = model.bn.running_mean
    cpu_var = model.bn.running_var

    model.bn.half()
    model.bn.running_mean = model.bn.running_mean.to(torch.float)
    model.bn.running_var = model.bn.running_var.to(torch.float)

    poptorch_model = poptorch.inferenceModel(model)
    _, ipu_var, ipu_mean = poptorch_model(fake_data)

    # We lose some precision in the half conversion.
    helpers.assert_allclose(actual=ipu_mean,
                            expected=cpu_mean.half(),
                            rtol=1e-02,
                            atol=1e-02)

    helpers.assert_allclose(actual=ipu_var,
                            expected=cpu_var.half(),
                            rtol=1e-02,
                            atol=1e-02)


def test_half_casts_outplace():
    torch.manual_seed(42)
    opts = poptorch.Options()

    class Model(torch.nn.Module):
        def forward(self, x1, x2):
            return x1, x2, x1.to(torch.float16), x2.half()

    model = Model()
    poptorch_model = poptorch.inferenceModel(model, opts)

    x1 = torch.tensor([0], dtype=torch.float32)
    x2 = torch.tensor([0], dtype=torch.float32)

    x1_ipu, x2_ipu, x1_cast, x2_cast = poptorch_model(x1, x2)
    assert x1_ipu.dtype == torch.float32
    assert x2_ipu.dtype == torch.float32
    assert x1_cast.dtype == torch.float16
    assert x2_cast.dtype == torch.float16


def test_8bit_io_casting():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
            x1 = self.linear(x.half())
            x2 = self.linear(x.to(torch.half))
            x3 = self.linear(x.float())
            x4 = self.linear(x.to(torch.float))
            return x1, x2, x3, x4

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    x = torch.tensor([0], dtype=torch.uint8)

    y = poptorch_model(x)
    assert y[0].dtype == torch.half
    assert y[1].dtype == torch.half
    assert y[2].dtype == torch.float
    assert y[3].dtype == torch.float


def test_buffers_without_parameters_can_be_traced():
    torch.manual_seed(0)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("b", torch.randn(3, 3))

        def forward(self, x):
            return torch.matmul(self.b, x)

    model = Model()
    model.half()
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_model(torch.randn(3, 3).half())
