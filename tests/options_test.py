#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import unittest.mock

import tempfile
import os
import torch
import torch.nn as nn
import pytest
import poptorch
import poptorch.testing
import helpers


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
    with tempfile.TemporaryDirectory() as tmp:
        opts.deviceIterations(1).setExecutionStrategy(
            poptorch.PipelinedExecution()).replicationFactor(1).logDir(
                tmp).enableSyntheticData(True)
    inference_model = poptorch.inferenceModel(model, opts)

    x = torch.ones(2)
    y = torch.zeros(2)

    inference_model(x, y)


@helpers.printCapfdOnExit
def test_set_options_from_file(capfd):
    poptorch.setLogLevel("DEBUG")  # Force debug logging

    class LogChecker(helpers.LogChecker):
        def validate(self):
            # pylint: disable=line-too-long
            self.assert_contains(
                "poptorch.Options set replication_factor to value 1")
            self.assert_contains(
                "poptorch.Options set device_iterations to value 1")
            self.assert_contains(
                "poptorch.Options set execution_mode to value 1")
            self.assert_contains(
                "poptorch.Options set syntheticDataMode to value 2")

    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    options_list = [
        "deviceIterations(1)",
        "setExecutionStrategy(poptorch.ShardedExecution())",
        "  replicationFactor(1)",  # Whitespace should be stripped
        " ",  # Empty lines should be skipped
        "enableSyntheticData(True) # Inline comments should be ignored",
        "# Comments should be ignored"
    ]
    options_list = "\n".join(options_list)

    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, "tmp.conf")
        f = open(filepath, "w")
        # Write the options to file
        f.write(options_list)
        f.close()

        opts = poptorch.Options()
        # Read the options back
        opts.loadFromFile(filepath)

        # Ensure that a useful error message is output on malformed input
        f = open(filepath, "a")
        f.write("\nanchorMode(poptorch.AnchorMode.All")
        f.close()
        with pytest.raises(poptorch.options.ConfigFileError) as e:
            opts.loadFromFile(filepath)
        assert "SyntaxError at line 5 of tmp.conf: unexpected EOF " \
               "while parsing\n" \
               "> options.anchorMode(poptorch.AnchorMode.All" in str(e.value)

    # Create the model
    model = Network()
    inference_model = poptorch.inferenceModel(model, opts)

    x = torch.ones(2)
    y = torch.zeros(2)

    # Run the model
    inference_model(x, y)

    testlog = LogChecker(capfd)
    # Ensure the options were actually set
    testlog.validate()


def test_set_popart_options():
    # pylint: disable=protected-access
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    # Create our model.
    model = Network()
    opts = poptorch.Options()
    opts._Popart.set("hardwareInstrumentations", set([0, 1]))
    opts._Popart.set("dotChecks", [0, 1])
    opts._Popart.set("engineOptions", {
        "debug.allowOutOfMemory": "true",
    })
    opts._Popart.set("customCodelets", [])
    opts._Popart.set("autoRecomputation", 1)
    opts._Popart.set("enableOutlining", True)
    opts._Popart.set("batchSerializationSettings.factor", 1)
    opts._Popart.set("batchSerializationSettings.concatOnVirtualGraphChange",
                     True)
    opts._Popart.set("batchSerializationSettings.concatOnExecutionPhaseChange",
                     True)
    opts._Popart.set("batchSerializationSettings.concatOnPipelineStageChange",
                     True)
    opts._Popart.set("batchSerializationSettings.transformContext", 0)
    opts._Popart.set("batchSerializationSettings.method", 0)
    opts._Popart.set("batchSerializationSettings.batchSchedule", 1)

    opts._Popart.set("accumulateOuterFragmentSettings.schedule", 1)
    opts._Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs",
                     ["0", "1"])

    inference_model = poptorch.inferenceModel(model, opts)
    x = torch.ones(2)
    y = torch.zeros(2)

    # The options above don't really make sense so check they're being passed
    # to the backend without causing any error but don't actually run the
    # model.
    inference_model.compile(x, y)


def test_popart_patterns():
    # pylint: disable=protected-access
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    # Create our model.
    model = Network()
    opts = poptorch.Options()
    patterns = {"PadSum": True}
    opts._Popart.setPatterns(patterns, 0)
    inference_model = poptorch.inferenceModel(model, opts)
    x = torch.ones(2)
    y = torch.zeros(2)

    inference_model(x, y)


@helpers.printCapfdOnExit
@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("ptype", [torch.half, torch.float])
def test_popart_partials(capfd, dtype, ptype):
    # pylint: disable=protected-access
    torch.manual_seed(42)
    x = torch.randn((1, 16, 16), dtype=dtype)

    model = torch.nn.Sequential()
    model.add_module('lin', torch.nn.Linear(16, 16))
    model.add_module('conv', torch.nn.Conv1d(16, 16, 1))

    poptorch.setLogLevel(0)
    opts = poptorch.Options()
    opts.Precision.setPartialsType(ptype)
    poptorch_model = poptorch.inferenceModel(model, opts)
    poptorch_model(x)

    log = helpers.LogChecker(capfd)
    if ptype == torch.float:
        log.assert_contains(
            'poptorch.Options set partialsTypeMatMuls to value float')
        log.assert_contains(
            'poptorch.Options set convolutionOptions[partialsType] to float')
        log.assert_contains('"partialsType":"MatMulPartialsType::FLOAT"')
        log.assert_contains('"partialsType[0]":"float"')
    else:
        log.assert_contains(
            'poptorch.Options set partialsTypeMatMuls to value half')
        log.assert_contains(
            'poptorch.Options set convolutionOptions[partialsType] to half')
        log.assert_contains('"partialsType":"MatMulPartialsType::HALF"')
        log.assert_contains('"partialsType[0]":"half"')


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


@unittest.mock.patch.dict("os.environ", helpers.disableAllModels())
def test_offline_ipu():
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    model = Network()
    # Force-disable the IPU model
    opts = poptorch.Options().useOfflineIpuTarget()
    poptorch.inferenceModel(model, opts)

    inference_model = poptorch.inferenceModel(model, opts)
    x = torch.ones(2)
    y = torch.zeros(2)

    with pytest.raises(AssertionError,
                       match="Trying to run a model on an offline device"):
        inference_model(x, y)


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


@helpers.printCapfdOnExit
@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("setting", [True, False, None])
def test_running_variance(capfd, dtype, setting):
    x = torch.randn((16, 16), dtype=dtype)

    model = torch.nn.Sequential()
    model.add_module('lin', torch.nn.Linear(16, 16))
    model.add_module('bn', torch.nn.BatchNorm1d(16))

    if dtype == torch.half:
        model.half()

    poptorch.setLogLevel(0)
    opts = poptorch.Options()
    if setting is not None:
        opts.Precision.runningVarianceAlwaysFloat(setting)
    poptorch_model = poptorch.inferenceModel(model, opts)
    poptorch_model(x)

    log = helpers.LogChecker(capfd)
    if setting is None or setting:
        log.assert_contains(
            "poptorch.Options set runningVarianceAlwaysFloat to true")
    else:
        log.assert_contains(
            "poptorch.Options set runningVarianceAlwaysFloat to false")

    if dtype == torch.float:
        log.assert_contains("%24 : Float(16:1, requires_grad=0, device=cpu)):")
    elif setting is None or setting:
        log.assert_contains("%24 : Float(16:1, requires_grad=0, device=cpu)):")
    else:
        log.assert_contains("%24 : Half(16:1, requires_grad=0, device=cpu)):")


def test_ipu_context_flag():
    class Network(nn.Module):
        def forward(self, x, y):
            if poptorch.isRunningOnIpu():
                output = x + y
            else:
                output = x * y

            return output

    model = Network()

    inference_model = poptorch.inferenceModel(model)

    x = torch.tensor([50])
    y = torch.tensor([2])

    assert inference_model(x, y) == 52
    assert model(x, y) == 100
