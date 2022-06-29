#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from io import StringIO
import json
import pytest
import torch
from torch import nn
import poptorch
import helpers


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
@pytest.mark.parametrize("trace_model", [True, False])
def test_multiconv_basic(num_layers, trace_model):
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

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(m, options)
    poptorch_out = poptorch_model(input)
    assert_contains_multiconv(poptorch_model, num_layers)

    for cpu, pop in zip(native, poptorch_out):
        helpers.assert_allclose(expected=cpu, actual=pop)


def multiconv_harness(multiconv, trace_model):
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
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(m, options)
    poptorch_out = poptorch_model(x)
    helpers.assert_allclose(expected=native, actual=poptorch_out)
    assert_contains_multiconv(poptorch_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_multiconv_options_broadcast(trace_model):
    multiconv = (
        poptorch.MultiConv().availableMemoryProportions(0.8).partialsTypes(
            torch.float).planType(
                poptorch.MultiConvPlanType.Parallel).perConvReservedTiles(
                    100).cycleBackOff(0.3)).enableConvDithering(True)

    multiconv_harness(multiconv, trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_multiconv_options_per_conv(trace_model):
    partials_types = [torch.float, torch.float]
    multiconv = (poptorch.MultiConv().availableMemoryProportions(
        (0.8, 0.7)).partialsTypes(partials_types).planType(
            poptorch.MultiConvPlanType.Parallel).perConvReservedTiles(
                120).cycleBackOff(0.4)).enableConvDithering(True)

    multiconv_harness(multiconv, trace_model)


@pytest.mark.parametrize("trace_model", [True, False])
def test_multiconv_layers(trace_model):
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
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    poptorch_out = poptorch_model(input)

    assert_contains_multiconv(poptorch_model)
    helpers.assert_allclose(actual=poptorch_out, expected=native_out)


@pytest.mark.parametrize("trace_model", [True, False])
def test_invalid_multiconv_nested(trace_model):
    if not trace_model:
        pytest.skip("TODO(T57195): Did not raise poptorch_core.Error")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 10, 10)

        def forward(self, x):
            with poptorch.MultiConv():
                with poptorch.MultiConv():
                    return self.conv(x)

    m = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(m, options)
    msg = "Nested poptorch.MultiConv is not supported"

    with pytest.raises(poptorch.Error, match=msg):
        poptorch_model(torch.zeros(2, 1, 32, 32))


@pytest.mark.parametrize("trace_model", [True, False])
def test_invalid_multiconv_empty(trace_model):
    if not trace_model:
        pytest.skip("TODO(T57195): Did not raise poptorch_core.Error")

    class Model(torch.nn.Module):
        def forward(self, x):
            with poptorch.MultiConv():
                return torch.pow(x, 2)

    m = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(m, options)
    msg = "Unexpected end_multi_conv"

    with pytest.raises(poptorch.Error, match=msg):
        poptorch_model(torch.ones(2, 2))


def test_invalid_multiconv_options():
    mc = poptorch.MultiConv()

    with pytest.raises(ValueError, match="Invalid partials types"):
        mc.partialsTypes("half")

    with pytest.raises(AssertionError, match="Invalid plan type"):
        mc.planType("parallel")
