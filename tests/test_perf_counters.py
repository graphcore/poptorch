# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import poptorch
import helpers


class Model(torch.nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)


def assert_perf_counter_size(perf, inputs, outputs, steps, outsteps=None):
    def assert_size(perf, elems, steps):
        assert len(perf) == elems
        for elem in perf:
            assert len(elem) == steps

    outsteps = outsteps or steps

    assert_size(perf['input'], inputs, steps)
    assert_size(perf['input_complete'], inputs, steps)
    assert_size(perf['output'], outputs, outsteps)
    assert_size(perf['output_complete'], outputs, outsteps)


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


@pytest.mark.parametrize("mode_tuple", [(poptorch.AnchorMode.Final, 1),
                                        (poptorch.AnchorMode.All, 1),
                                        (poptorch.AnchorMode.Sum, 1),
                                        (poptorch.AnchorMode.EveryN, 2)])
@pytest.mark.parametrize("steps", [2, 4])
@pytest.mark.parametrize("replicas", [1, 2])
@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_inference(mode_tuple, steps, replicas):
    model = Model()
    opts = poptorch.Options()
    opts.anchorMode(mode_tuple[0], mode_tuple[1])
    opts.deviceIterations(steps)
    opts.replicationFactor(replicas)
    poptorch_model = poptorch.inferenceModel(model, opts)

    torch.manual_seed(42)
    x = torch.randn(16, 100, 100)
    y = torch.randn(16, 100, 100)
    poptorch_model(x, y)
    perf = poptorch_model.getPerfCounters()

    outsteps = steps * replicas
    if mode_tuple[0] in [poptorch.AnchorMode.Final, poptorch.AnchorMode.Sum]:
        outsteps = replicas
    elif mode_tuple[0] is poptorch.AnchorMode.EveryN:
        outsteps = steps // mode_tuple[1] * replicas
    assert_perf_counter_size(perf, 2, 1, steps * replicas, outsteps)

    latency = poptorch_model.getLatency()
    assert_latency_values(latency)


@pytest.mark.parametrize("mode_tuple", [(poptorch.AnchorMode.Final, 1),
                                        (poptorch.AnchorMode.All, 1),
                                        (poptorch.AnchorMode.Sum, 1),
                                        (poptorch.AnchorMode.EveryN, 2)])
@pytest.mark.parametrize("steps", [2, 4])
@pytest.mark.parametrize("accums", [1, 2])
@pytest.mark.parametrize("replicas", [1, 2])
@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_training(mode_tuple, steps, accums, replicas):
    torch.manual_seed(42)
    inputs = torch.randn(16, 100)
    targets = torch.randn(16, 100)

    opts = poptorch.Options()
    opts.anchorMode(mode_tuple[0], mode_tuple[1])
    opts.deviceIterations(steps)
    opts.Training.gradientAccumulation(accums)
    opts.replicationFactor(replicas)

    model = torch.nn.Linear(100, 100)
    poptorch_model = helpers.trainingModelWithLoss(model,
                                                   loss=torch.nn.L1Loss(),
                                                   options=opts)

    poptorch_model(inputs, targets)
    perf = poptorch_model.getPerfCounters()

    outsteps = steps * accums * replicas
    if mode_tuple[0] in [poptorch.AnchorMode.Final, poptorch.AnchorMode.Sum]:
        outsteps = replicas
    elif mode_tuple[0] is poptorch.AnchorMode.EveryN:
        outsteps = steps // mode_tuple[1] * accums * replicas

    assert_perf_counter_size(perf, 2, 2, steps * accums * replicas, outsteps)

    latency = poptorch_model.getLatency()
    assert_latency_values(latency)
