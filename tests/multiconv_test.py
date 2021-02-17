#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from io import StringIO
import json
import pytest
import torch
from torch import nn
import poptorch


def getPopartMultiConvs(poptorch_model):
    ir_as_json = json.load(StringIO(poptorch_model._debugGetPopartIR()))  # pylint: disable=protected-access
    assert "maingraph" in ir_as_json, "Expected maingraph in serialized IR."

    r = []
    for op in ir_as_json["maingraph"]:
        if op["type"] == "MultiConv":
            r.append(op)

    return r


def assert_contains_multiconv(poptorch_model, expected_num=1):
    num_multiconv = len(getPopartMultiConvs(poptorch_model))
    msg = (f"Wrong number of MultiConv ops.\n"
           f"   Expected : {expected_num}\n"
           f"   Actual   : {num_multiconv}.")
    assert num_multiconv == expected_num, msg


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_multiconv_basic(num_layers):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.convA = nn.Conv2d(1, 1, 5)
            self.convB = nn.Conv2d(1, 1, 5, bias=False)

        def forward(self, x):
            with poptorch.MultiConv():
                a = self.convA(x)
                absx = torch.abs(x)
                b = self.convB(absx)
                return a + b

    m = [Model() for i in range(num_layers)]
    m = torch.nn.Sequential(*m)
    torch.manual_seed(0)
    input = torch.randn(2, 1, 28, 28)

    native = m(input)

    poptorch_model = poptorch.inferenceModel(m)
    poptorch_out = poptorch_model(input)
    assert_contains_multiconv(poptorch_model, num_layers)

    for cpu, pop in zip(native, poptorch_out):
        torch.testing.assert_allclose(cpu, pop)


def multiconv_harness(multiconv):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 10, 5)
            self.conv2 = nn.Conv2d(1, 10, 5)
            self.MultiConv = multiconv

        def forward(self, x):
            y = torch.pow(x, 2)

            with self.MultiConv:
                u = self.conv1(x)
                v = self.conv2(y)

            return u - v

    m = Model()
    torch.manual_seed(0)
    x = torch.randn(2, 1, 28, 28)

    native = m(x)
    poptorch_model = poptorch.inferenceModel(m)
    poptorch_out = poptorch_model(x)
    torch.testing.assert_allclose(native, poptorch_out)
    assert_contains_multiconv(poptorch_model)


def test_multiconv_options_broadcast_deprecated():
    multiconv = (
        poptorch.MultiConv().availableMemoryProportions(0.8).partialsTypes(
            poptorch.MultiConvPartialsType.Float).planType(
                poptorch.MultiConvPlanType.Parallel).perConvReservedTiles(
                    100).cycleBackOff(0.3))

    multiconv_harness(multiconv)


def test_multiconv_options_broadcast():
    multiconv = (
        poptorch.MultiConv().availableMemoryProportions(0.8).partialsTypes(
            torch.float).planType(
                poptorch.MultiConvPlanType.Parallel).perConvReservedTiles(
                    100).cycleBackOff(0.3))

    multiconv_harness(multiconv)


def test_multiconv_options_per_conv_deprecated():
    partials_types = [
        poptorch.MultiConvPartialsType.Float,
        poptorch.MultiConvPartialsType.Float
    ]
    multiconv = (poptorch.MultiConv().availableMemoryProportions(
        (0.8, 0.7)).partialsTypes(partials_types).planType(
            poptorch.MultiConvPlanType.Parallel).perConvReservedTiles(
                120).cycleBackOff(0.4))

    multiconv_harness(multiconv)


def test_multiconv_options_per_conv():
    partials_types = [torch.float, torch.float]
    multiconv = (poptorch.MultiConv().availableMemoryProportions(
        (0.8, 0.7)).partialsTypes(partials_types).planType(
            poptorch.MultiConvPlanType.Parallel).perConvReservedTiles(
                120).cycleBackOff(0.4))

    multiconv_harness(multiconv)


def test_multiconv_layers():
    class Network(nn.Module):
        def __init__(self):
            super().__init__()

            self.layer1A = nn.Sequential(nn.Conv2d(1, 10, 5), nn.MaxPool2d(2),
                                         nn.ReLU())
            self.layer1B = nn.Sequential(nn.Conv2d(1, 10, 5), nn.MaxPool2d(2),
                                         nn.ReLU())
            self.layer2 = nn.Sequential(nn.Conv2d(10, 20, 5), nn.MaxPool2d(2),
                                        nn.ReLU())
            self.layer3 = nn.Linear(320, 256)
            self.layer3_act = nn.ReLU()
            self.layer4 = nn.Linear(256, 10)

            self.softmax = nn.LogSoftmax(1)

        def forward(self, x):
            with poptorch.MultiConv():
                absx = torch.abs(x)
                y = self.layer1A(absx)
                z = self.layer1B(x)
                x = y + z

            x = self.layer2(x)
            x = x.view(-1, 320)
            x = self.layer3_act(self.layer3(x))
            x = self.layer4(x)
            x = self.softmax(x)
            return x

    model = Network()
    # Run on CPU.
    input = torch.randn(2, 1, 28, 28)
    native_out = model(input)
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_out = poptorch_model(input)

    assert_contains_multiconv(poptorch_model)
    torch.testing.assert_allclose(poptorch_out, native_out)


def test_invalid_multiconv_nested():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 10, 10)

        def forward(self, x):
            with poptorch.MultiConv():
                with poptorch.MultiConv():
                    return self.conv(x)

    m = Model()
    poptorch_model = poptorch.inferenceModel(m)
    msg = "Nested poptorch.MultiConv is not supported"

    with pytest.raises(RuntimeError, match=msg):
        poptorch_model(torch.zeros(2, 1, 32, 32))


def test_invalid_multiconv_empty():
    class Model(torch.nn.Module):
        def forward(self, x):
            with poptorch.MultiConv():
                return torch.pow(x, 2)

    m = Model()
    poptorch_model = poptorch.inferenceModel(m)
    msg = "Unexpected end_multi_conv"

    with pytest.raises(RuntimeError, match=msg):
        poptorch_model(torch.ones(2, 2))


def test_invalid_multiconv_options():
    mc = poptorch.MultiConv()

    with pytest.raises(ValueError, match="Invalid partials types"):
        mc.partialsTypes("half")

    with pytest.raises(AssertionError, match="Invalid plan type"):
        mc.planType("parallel")
