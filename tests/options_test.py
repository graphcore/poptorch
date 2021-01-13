#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import unittest.mock

import torch
import torch.nn as nn
import pytest
import poptorch
import poptorch.testing


def test_jit_script():
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    # Create our model.
    model = Network()
    opts = poptorch.Options()
    opts.Jit.traceModel(False)
    inference_model = poptorch.inferenceModel(model, opts)

    x = torch.ones(2)
    y = torch.zeros(2)

    ipu = inference_model(x, y)
    ref = model(x, y)
    assert poptorch.testing.allclose(
        ref, ipu), "%s doesn't match the expected output %s" % (ipu, ref)


def test_set_options():
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    # Create our model.
    model = Network()
    opts = poptorch.Options()
    # Just set a bunch of options and check they're successfully parsed.
    opts.deviceIterations(1).setExecutionStrategy(
        poptorch.PipelinedExecution()).replicationFactor(1).logDir("/tmp")
    inference_model = poptorch.inferenceModel(model, opts)

    x = torch.ones(2)
    y = torch.zeros(2)

    inference_model(x, y)


def test_set_popart_options():
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    # Create our model.
    model = Network()
    opts = poptorch.Options()
    opts.Popart.set("hardwareInstrumentations", set([0, 1]))
    opts.Popart.set("dotChecks", [0, 1])
    opts.Popart.set("engineOptions", {
        "debug.allowOutOfMemory": "true",
        "exchange.streamBufferOverlap": "any"
    })
    opts.Popart.set("customCodelets", [])
    opts.Popart.set("autoRecomputation", 1)
    opts.Popart.set("cachePath", "/tmp")
    opts.Popart.set("enableOutlining", True)
    opts.Popart.set("batchSerializationSettings.factor", 1)
    opts.Popart.set("batchSerializationSettings.concatOnVirtualGraphChange",
                    True)
    opts.Popart.set("batchSerializationSettings.concatOnExecutionPhaseChange",
                    True)
    opts.Popart.set("batchSerializationSettings.concatOnPipelineStageChange",
                    True)
    opts.Popart.set("batchSerializationSettings.transformContext", 0)
    opts.Popart.set("batchSerializationSettings.method", 0)
    opts.Popart.set("batchSerializationSettings.batchSchedule", 1)

    opts.Popart.set("accumulateOuterFragmentSettings.schedule", 1)
    opts.Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs",
                    ["0", "1"])

    inference_model = poptorch.inferenceModel(model, opts)
    x = torch.ones(2)
    y = torch.zeros(2)

    inference_model(x, y)


def test_popart_patterns():
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    # Create our model.
    model = Network()
    opts = poptorch.Options()
    patterns = {"PadSum": True}
    opts.Popart.setPatterns(patterns, 0)
    inference_model = poptorch.inferenceModel(model, opts)
    x = torch.ones(2)
    y = torch.zeros(2)

    inference_model(x, y)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_real_ipu_selection():
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    model = Network()
    # Force-disable the IPU model
    opts = poptorch.Options().useIpuModel(False)
    inference_model = poptorch.inferenceModel(model, opts)
    x = torch.ones(2)
    y = torch.zeros(2)

    inference_model(x, y)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_ipu_id_selection():
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    model = Network()
    # Force-disable the IPU model
    opts = poptorch.Options().useIpuId(0)
    inference_model = poptorch.inferenceModel(model, opts)
    x = torch.ones(2)
    y = torch.zeros(2)

    inference_model(x, y)


@unittest.mock.patch.dict("os.environ", {"POPTORCH_IPU_MODEL": "0"})
def test_offline_ipu():
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    model = Network()
    # Force-disable the IPU model
    opts = poptorch.Options().useOfflineIpuTarget()
    poptorch.inferenceModel(model, opts)

    #TODO(T23447): Support offline compilation
    #inference_model = poptorch.inferenceModel(model, opts)
    #x = torch.ones(2)
    #y = torch.zeros(2)

    #ipu = inference_model(x, y)
    #assert not ipu, "Offline compilation shouldn't return anything"


def test_tensor_location():
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    model = Network()
    opts = poptorch.Options()
    opts.TensorLocations.setActivationLocation(
        poptorch.TensorLocationSettings().minElementsForOffChip(
            4).useOnChipStorage(True))
    opts.TensorLocations.setWeightLocation(
        poptorch.TensorLocationSettings().useIOTilesToStore(
            True).useReplicatedTensorSharding(False))
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings().useIOTilesToLoad(
            False).useReplicatedTensorSharding(
                True).minElementsForReplicatedTensorSharding(4))
    opts.TensorLocations.setAccumulatorLocation(
        poptorch.TensorLocationSettings().useOnChipStorage(False))
    inference_model = poptorch.inferenceModel(model, opts)
    x = torch.ones(2)
    y = torch.zeros(2)

    inference_model(x, y)
