#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import pytest
import torch
import torch.nn as nn
import helpers
import poptorch


@pytest.mark.parametrize("trace_model", [True, False])
def test_multiple_tensors(trace_model):
    class Network(nn.Module):
        def forward(self, x, y):

            t1 = (x + y)
            t2 = (t1, x * y)

            return t2[0], y - x, t2[1] + t1

    # Create our model.
    model = Network()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    x = torch.ones(2)
    y = torch.zeros(2)

    ipu = inference_model(x, y)
    ref = model(x, y)
    helpers.assert_allclose(actual=ipu, expected=ref)


@pytest.mark.parametrize("trace_model", [True, False])
def test_simple_list(trace_model):
    class Network(nn.Module):
        def forward(self, x, y):

            t1 = (x + y)
            t2 = (t1, x * y)

            return [t2[0], y - x, t2[1] + t1]

    # Create our model.
    model = Network()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    x = torch.ones(2)
    y = torch.zeros(2)

    ipu = inference_model(x, y)
    ref = model(x, y)
    helpers.assert_allclose(actual=ipu, expected=ref)


@pytest.mark.parametrize("trace_model", [True, False])
def test_simple_tuple(trace_model):
    class Network(nn.Module):
        def forward(self, x, y):

            t1 = (x + y)
            t2 = (t1, x * y)

            return (t2[0], y - x, t2[1] + t1)

    # Create our model.
    model = Network()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    x = torch.ones(2)
    y = torch.zeros(2)

    ipu = inference_model(x, y)
    ref = model(x, y)
    helpers.assert_allclose(actual=ipu, expected=ref)


@pytest.mark.parametrize("trace_model", [True, False])
def test_nested_tuples(trace_model):
    class Network(nn.Module):
        def forward(self, x, y):

            t1 = (x + y)
            t2 = (t1, x * y)

            return x, (t2, y - x, t2[1] + t1), (y, ((t1 * 2.0)))

    # Create our model.
    model = Network()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    x = torch.ones(2)
    y = torch.zeros(2)

    ipu = inference_model(x, y)
    ref = model(x, y)

    helpers.assert_allclose(actual=ipu, expected=ref)


@pytest.mark.parametrize("trace_model", [True, False])
def test_same_tensor(trace_model):
    class Network(nn.Module):
        def forward(self, x, y):

            t1 = (x + y)
            t2 = (t1, x * y)

            return t1, (t1, t2, t1)

    # Create our model.
    model = Network()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    x = torch.ones(2)
    y = torch.zeros(2)

    ipu = inference_model(x, y)
    ref = model(x, y)

    helpers.assert_allclose(actual=ipu, expected=ref)


def test_dict():
    class Network(nn.Module):
        def forward(self, x, y):

            t1 = (x + y)
            t2 = (x * y)

            # Note: keys are not in alphabetical order
            return {'b': t1, 'a': t2}

    # Create our model.
    cpu_model = Network()
    options = poptorch.Options()
    options.Jit.traceModel(False)
    ipu_model = poptorch.inferenceModel(cpu_model, options)

    x = torch.ones(2)
    y = torch.zeros(2)

    cpu_res = cpu_model(x, y)
    ipu_res = ipu_model(x, y)

    # Check the outputs are the same
    assert cpu_res.keys() == ipu_res.keys()
    for k in cpu_res.keys():
        assert torch.allclose(cpu_res[k], ipu_res[k])
