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


def assert_latency_values(model):
    def check(latency):
        (minimum, maximum, average) = latency
        assert minimum <= average
        assert average <= maximum

    host2ipu = model.getHostIpuLatency()
    compute = model.getComputeLatency()
    ipu2host = model.getIpuHostLatency()
    round_trip = model.getLatency()

    check(host2ipu)
    check(compute)
    check(ipu2host)
    check(round_trip)


def test_simple():
    x = torch.randn(100, 100)
    y = torch.randn(100, 100)
    model = Model()
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_model(x, y)

    perf = poptorch_model.getPerfCounters()
    assert_perf_counter_size(perf, 2, 1, 1)
    assert_latency_values(poptorch_model)


def test_steps():
    x = torch.randn(10, 100, 100)
    y = torch.randn(10, 100, 100)
    model = Model()
    opts = poptorch.Options().deviceIterations(10)
    poptorch_model = poptorch.inferenceModel(model, opts)
    poptorch_model(x, y)

    perf = poptorch_model.getPerfCounters()
    assert_perf_counter_size(perf, 2, 1, 10)
    assert_latency_values(poptorch_model)


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
    assert_latency_values(poptorch_model)


@pytest.mark.parametrize("mode_tuple", [(poptorch.OutputMode.Final, 1),
                                        (poptorch.OutputMode.All, 1),
                                        (poptorch.OutputMode.Sum, 1),
                                        (poptorch.OutputMode.EveryN, 2)])
@pytest.mark.parametrize("steps", [2, 4])
@pytest.mark.parametrize("replicas", [1, 2])
@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_inference(mode_tuple, steps, replicas):
    model = Model()
    opts = poptorch.Options()
    opts.outputMode(mode_tuple[0], mode_tuple[1])
    opts.deviceIterations(steps)
    opts.replicationFactor(replicas)
    poptorch_model = poptorch.inferenceModel(model, opts)

    torch.manual_seed(42)
    x = torch.randn(16, 100, 100)
    y = torch.randn(16, 100, 100)
    poptorch_model(x, y)
    perf = poptorch_model.getPerfCounters()

    outsteps = steps * replicas
    if mode_tuple[0] in [poptorch.OutputMode.Final, poptorch.OutputMode.Sum]:
        outsteps = replicas
    elif mode_tuple[0] is poptorch.OutputMode.EveryN:
        outsteps = steps // mode_tuple[1] * replicas
    assert_perf_counter_size(perf, 2, 1, steps * replicas, outsteps)
    assert_latency_values(poptorch_model)


@pytest.mark.parametrize("mode_tuple", [(poptorch.OutputMode.Final, 1),
                                        (poptorch.OutputMode.All, 1),
                                        (poptorch.OutputMode.Sum, 1),
                                        (poptorch.OutputMode.EveryN, 2)])
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
    opts.outputMode(mode_tuple[0], mode_tuple[1])
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
    if mode_tuple[0] in [poptorch.OutputMode.Final, poptorch.OutputMode.Sum]:
        outsteps = replicas
    elif mode_tuple[0] is poptorch.OutputMode.EveryN:
        outsteps = steps // mode_tuple[1] * accums * replicas

    assert_perf_counter_size(perf, 2, 2, steps * accums * replicas, outsteps)
    assert_latency_values(poptorch_model)


def test_synthetic_data():
    model = Model()
    opts = poptorch.Options()
    opts.deviceIterations(16)
    opts.enableSyntheticData(True)
    poptorch_model = poptorch.inferenceModel(model, opts)

    torch.manual_seed(42)
    x = torch.randn(16, 100, 100)
    y = torch.randn(16, 100, 100)
    poptorch_model(x, y)
    perf = poptorch_model.getPerfCounters()

    assert_perf_counter_size(perf, 2, 1, 0, 0)

    latency = poptorch_model.getLatency()
    assert latency == (0., 0., 0.)
