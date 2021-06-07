# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import poptorch


class Model(torch.nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)


def assert_perf_counter_size(perf, inputs, outputs, steps):
    def assert_size(perf, elems, steps):
        assert len(perf) == elems
        for elem in perf:
            assert len(elem) == steps

    assert_size(perf['input'], inputs, steps)
    assert_size(perf['input_complete'], inputs, steps)
    assert_size(perf['output'], outputs, steps)
    assert_size(perf['output_complete'], outputs, steps)


def assert_latency_values(latency):
    (minimum, maximum, average) = latency
    assert minimum <= average
    assert average <= maximum


def test_simple():
    x = torch.randn(100, 100)
    y = torch.randn(100, 100)
    model = Model()
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_model(x, y)

    perf = poptorch_model.getPerfCounters()
    assert_perf_counter_size(perf, 2, 1, 1)

    latency = poptorch_model.getLatency()
    assert_latency_values(latency)


def test_steps():
    x = torch.randn(10, 100, 100)
    y = torch.randn(10, 100, 100)
    model = Model()
    opts = poptorch.Options().deviceIterations(10)
    poptorch_model = poptorch.inferenceModel(model, opts)
    poptorch_model(x, y)

    perf = poptorch_model.getPerfCounters()
    assert_perf_counter_size(perf, 2, 1, 10)

    latency = poptorch_model.getLatency()
    assert_latency_values(latency)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_replicas():
    x = torch.randn(4, 100, 100)
    y = torch.randn(4, 100, 100)
    model = Model()
    opts = poptorch.Options().replicationFactor(4)
    poptorch_model = poptorch.inferenceModel(model, opts)
    poptorch_model(x, y)

    perf = poptorch_model.getPerfCounters()
    assert_perf_counter_size(perf, 2, 1, 4)

    latency = poptorch_model.getLatency()

    assert_latency_values(latency)
