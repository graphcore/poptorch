#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import unittest.mock
import tempfile
import os
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import pytest
import poptorch
from poptorch.enums import MeanReductionStrategy
import helpers


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
@helpers.overridePoptorchLogLevel("DEBUG")
def test_set_options_from_file(capfd):
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
@helpers.overridePoptorchLogLevel("TRACE")
def test_popart_partials(capfd, dtype, ptype):
    # pylint: disable=protected-access
    torch.manual_seed(42)
    x = torch.randn((1, 16, 16), dtype=dtype)

    model = torch.nn.Sequential()
    model.add_module('lin', torch.nn.Linear(16, 16))
    model.add_module('conv', torch.nn.Conv1d(16, 16, 1))

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


@helpers.printCapfdOnExit
@pytest.mark.parametrize("optim", [
    optim.SGD,
    optim.Adam,
    optim.AdamW,
    optim.RMSprop,
    poptorch.optim.SGD,
    poptorch.optim.Adam,
    poptorch.optim.AdamW,
    poptorch.optim.RMSprop,
    poptorch.optim.LAMB,
])
@helpers.overridePoptorchLogLevel("DEBUG")
@helpers.printCapfdOnExit
def test_automatic_loss_scaling(capfd, optim):

    # Just a simple model with weights and a loss function
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(5, 2)

        def forward(self, x):
            x = self.lin(x)
            return poptorch.identity_loss(x, reduction='mean')

    # Create our model.
    model = Model()
    opts = poptorch.Options()
    opts.Training.setAutomaticLossScaling(True)

    # The lr value doesn't matter here, we just want to ensure the option is set
    if optim == poptorch.optim.SGD:
        optimizer = optim(model.parameters(), lr=0.0, use_combined_accum=False)
    else:
        optimizer = optim(model.parameters(), lr=0.0)
    training_model = poptorch.trainingModel(model, opts, optimizer)

    training_model(torch.ones(5))

    log = helpers.LogChecker(capfd)
    log.assert_contains(
        'poptorch.Options set automaticLossScalingSettings.enabled '
        'to value true')


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
@helpers.overridePoptorchLogLevel("TRACE")
def test_running_statistics(capfd, dtype, setting):
    x = torch.randn((16, 16), dtype=dtype)

    model = torch.nn.Sequential()
    model.add_module('lin', torch.nn.Linear(16, 16))
    model.add_module('bn', torch.nn.BatchNorm1d(16))

    if dtype == torch.half:
        model.half()

    opts = poptorch.Options()
    if setting is not None:
        opts.Precision.runningStatisticsAlwaysFloat(setting)
    poptorch_model = poptorch.inferenceModel(model, opts)
    poptorch_model(x)

    log = helpers.LogChecker(capfd)
    if setting is None or setting:
        log.assert_contains(
            "poptorch.Options set runningStatisticsAlwaysFloat to true")
    else:
        log.assert_contains(
            "poptorch.Options set runningStatisticsAlwaysFloat to false")

    dtype_str = "Float" if dtype == torch.float or \
        setting is None or setting else "Half"

    log.assert_contains(
        f"%22 : {dtype_str}(16, strides=[1], requires_grad=0, device=cpu)):")


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


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
@pytest.mark.parametrize("enabled", [True, False, None])
@helpers.overridePoptorchLogLevel("INFO")
def test_ipu_model(enabled, capfd):
    class Model(nn.Module):
        def forward(self, x, y):
            return x + y

    model = Model()
    opts = poptorch.Options()
    if enabled is not None:
        opts.useIpuModel(enabled)

    poptorch_model = poptorch.inferenceModel(model, opts)
    x = torch.tensor([50])
    y = torch.tensor([2])

    poptorch_model(x, y)

    log = helpers.LogChecker(capfd)
    if enabled is None:
        log.assert_not_contains("From the user configuration: Ipu model")
    elif enabled:
        log.assert_contains("From the user configuration: Ipu model: Enabled")
    else:
        log.assert_contains("From the user configuration: Ipu model: Disabled")


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed to count IPU cycles")
@helpers.overridePoptorchLogLevel("DEBUG")
def test_log_cycle_count(capfd):
    class LogChecker(helpers.LogChecker):
        def validate(self):
            self.assert_contains("Total number of IPU cycles: ")

    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    opts = poptorch.Options().logCycleCount(True)
    inference_model = poptorch.inferenceModel(Network(), opts)

    x = torch.tensor([1])
    y = torch.tensor([2])

    inference_model(x, y)

    assert inference_model.cycleCount() > 0

    log = LogChecker(capfd)
    log.validate()


def test_profile_report_with_model_name():
    def test(dirname):
        model = torch.nn.Linear(100, 100)
        opts = poptorch.Options()
        opts.modelName("tommyflowers")
        opts.enableProfiling(dirname)

        poptorch_model = poptorch.inferenceModel(model, opts)
        x = torch.randn(100, 100)
        poptorch_model(x)

    dirname = tempfile.mkdtemp()
    x = threading.Thread(target=test, args=(dirname, ))
    x.start()
    x.join()

    assert os.path.exists(os.path.join(dirname, "tommyflowers", "profile.pop"))


def test_profile_report():
    def test(dirname):
        model = torch.nn.Linear(100, 100)
        opts = poptorch.Options()
        opts.enableProfiling(dirname)

        poptorch_model = poptorch.inferenceModel(model, opts)
        x = torch.randn(100, 100)
        poptorch_model(x)

    dirname = tempfile.mkdtemp()
    x = threading.Thread(target=test, args=(dirname, ))
    x.start()
    x.join()

    assert os.path.exists(os.path.join(dirname, "inference", "profile.pop"))


mean_reduction_strategy_params = [
    # accum_type, training, combined_accum, correct_strategy

    # Post should be the float32 default
    (torch.float32, True, False, MeanReductionStrategy.Post),
    # Running should be the float16 default
    (torch.float16, True, False, MeanReductionStrategy.Running),
    # Running is not supported for combined_accum, so Post should be used
    (torch.float16, True, True, MeanReductionStrategy.Post),
    # The default accum_type is float32 so strategy should be Post when this is None
    (None, True, False, MeanReductionStrategy.Post),
    # The option isn't used in inference so it should remain as Post by default
    (None, False, False, MeanReductionStrategy.Post),
]


@pytest.mark.parametrize("params", mean_reduction_strategy_params)
def test_mean_reduction_strategy_implicit(params):
    accum_type, training, combined_accum, correct_strategy = params
    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])

    # A simple adder model just to test the correct strategy is set
    model = helpers.ModelWithWeights(lambda x, y: x + y, t1.shape)
    options = poptorch.Options()
    optimizer = poptorch.optim.SGD(model.parameters(),
                                   lr=0.01,
                                   accum_type=accum_type,
                                   use_combined_accum=combined_accum)

    poptorch_model = poptorch.trainingModel(
        model, options, optimizer) if training else poptorch.inferenceModel(
            model, options)

    poptorch_model.compile((t1, t2))

    assert (getattr(
        options.Training,
        "meanAccumulationAndReplicationReductionStrategy") == correct_strategy)


def test_mean_reduction_strategy_explicit():
    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])

    # A simple adder model just to test the correct strategy is set
    model = helpers.ModelWithWeights(lambda x, y: x + y, t1.shape)

    options = poptorch.Options()
    options.Training.setMeanAccumulationAndReplicationReductionStrategy(
        MeanReductionStrategy.Running)
    poptorch_model = poptorch.trainingModel(model, options)

    poptorch_model.compile((t1, t2))

    assert (getattr(options.Training,
                    "meanAccumulationAndReplicationReductionStrategy") ==
            MeanReductionStrategy.Running)


def test_num_io_tiles():
    options = poptorch.Options()

    error_msg = "numIOTiles must be an even number between 32 and 192."
    with pytest.raises(AssertionError, match=error_msg):
        options.TensorLocations.numIOTiles(10)
    with pytest.raises(AssertionError, match=error_msg):
        options.TensorLocations.numIOTiles(193)
    with pytest.raises(AssertionError, match=error_msg):
        options.TensorLocations.numIOTiles(33)

    options.TensorLocations.numIOTiles(32)
    options.TensorLocations.numIOTiles(192)
    options.TensorLocations.numIOTiles(100)
