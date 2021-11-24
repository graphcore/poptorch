#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import copy
import unittest.mock
import tempfile
import os
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import pytest
import poptorch
from poptorch.enums import OutputMode, HalfFloatCastingBehavior, MeanReductionStrategy
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
        f.write("\noutputMode(poptorch.OutputMode.All")
        f.close()
        with pytest.raises(poptorch.options.ConfigFileError) as e:
            opts.loadFromFile(filepath)
        assert "SyntaxError at line 5 of tmp.conf: unexpected EOF " \
               "while parsing\n" \
               "> options.outputMode(poptorch.OutputMode.All" in str(e.value)

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


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
def test_set_popart_options(capfd):
    # pylint: disable=protected-access
    class Network(nn.Module):
        def forward(self, x, y):
            return x + y

    # Create our model.
    model = Network()
    opts = poptorch.Options()
    opts._Popart.set("hardwareInstrumentations", set([0, 1]))
    opts._Popart.set("dotChecks", ["FINAL", "ALL"])
    opts._Popart.set("engineOptions", {
        "debug.allowOutOfMemory": "true",
    })
    opts._Popart.set("reportOptions", {"reportOptA": "A", "reportOptB": "B"})
    opts._Popart.set("convolutionOptions", {"convOptA": "A", "convOptB": "B"})
    opts._Popart.set("matmulOptions", {"matOptA": "A", "matOptB": "B"})
    opts._Popart.set("lstmOptions", {"lstmOptA": "A", "lstmOptB": "B"})
    opts._Popart.set("gclOptions", {"gclOptA": "A", "gclOptB": "B"})
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
    with pytest.raises(poptorch.Error, match=r"is not a valid .* option"):
        inference_model.compile(x, y)

    log = helpers.LogChecker(capfd)
    log.assert_contains("poptorch.Options added 0 to hardwareInstrumentations")
    log.assert_contains("poptorch.Options added 1 to hardwareInstrumentations")
    log.assert_contains("poptorch.Options added FINAL to dotChecks")
    log.assert_contains("poptorch.Options added ALL to dotChecks")
    log.assert_contains(
        "poptorch.Options set engineOptions[debug.allowOutOfMemory] to true")
    log.assert_contains("poptorch.Options set reportOptions[reportOptA] to A")
    log.assert_contains("poptorch.Options set reportOptions[reportOptB] to B")
    log.assert_contains(
        "poptorch.Options set convolutionOptions[convOptA] to A")
    log.assert_contains(
        "poptorch.Options set convolutionOptions[convOptB] to B")
    log.assert_contains("poptorch.Options set matmulOptions[matOptA] to A")
    log.assert_contains("poptorch.Options set matmulOptions[matOptB] to B")
    log.assert_contains("poptorch.Options set lstmOptions[lstmOptA] to A")
    log.assert_contains("poptorch.Options set lstmOptions[lstmOptB] to B")
    log.assert_contains("poptorch.Options set gclOptions[gclOptA] to A")
    log.assert_contains("poptorch.Options set gclOptions[gclOptB] to B")
    log.assert_contains("poptorch.Options set autoRecomputation to value 1")
    log.assert_contains("poptorch.Options set enableOutlining to value true")
    log.assert_contains(
        "poptorch.Options set batchSerializationSettings.factor to value 1")
    log.assert_contains(
        "poptorch.Options set "
        "batchSerializationSettings.concatOnVirtualGraphChange to value true")
    log.assert_contains(
        "poptorch.Options set "
        "batchSerializationSettings.concatOnExecutionPhaseChange to value true"
    )
    log.assert_contains(
        "poptorch.Options set "
        "batchSerializationSettings.concatOnPipelineStageChange to value true")
    log.assert_contains(
        "poptorch.Options set "
        "batchSerializationSettings.transformContext to value 0")
    log.assert_contains(
        "poptorch.Options set batchSerializationSettings.method to value 0")
    log.assert_contains(
        "poptorch.Options set batchSerializationSettings.batchSchedule "
        "to value 1")
    log.assert_contains(
        "poptorch.Options set accumulateOuterFragmentSettings.schedule "
        "to value 1")
    log.assert_contains(
        "poptorch.Options added 0 to "
        "accumulateOuterFragmentSettings.excludedVirtualGraphs")
    log.assert_contains(
        "poptorch.Options added 1 to "
        "accumulateOuterFragmentSettings.excludedVirtualGraphs")


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
    input = torch.ones(5)
    # Just a simple model with weights and a loss function
    model = helpers.ModelWithWeights(lambda x: x, input.shape)
    opts = poptorch.Options()
    opts.Training.setAutomaticLossScaling(True)

    # The lr value doesn't matter here, we just want to ensure the option is set
    if optim == poptorch.optim.SGD:
        optimizer = optim(model.parameters(), lr=0.0, use_combined_accum=False)
    else:
        optimizer = optim(model.parameters(), lr=0.0)
    training_model = poptorch.trainingModel(model, opts, optimizer)

    training_model((input, ))

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
        f"%24 : {dtype_str}(16, strides=[1], requires_grad=0, device=cpu)):")


def test_copying_options():
    # pylint: disable=protected-access
    opts = poptorch.Options()
    locationOnChip = poptorch.TensorLocationSettings()
    locationOnChip.useOnChipStorage(True)
    locationOutsideChip = poptorch.TensorLocationSettings()
    locationOutsideChip.useOnChipStorage(False)

    opts.deviceIterations(5)
    opts.Distributed.configureProcessId(5, 15)
    opts.anchorTensor("t1", "tensor1", OutputMode.EveryN, 2)
    opts._Popart.set("autoRecomputation", 3)
    opts._Popart.set("dummyKey", 5)
    opts.Training.gradientAccumulation(4)
    opts.TensorLocations.setWeightLocation(locationOnChip)
    opts.Precision.halfFloatCasting(HalfFloatCastingBehavior.HalfUpcastToFloat)
    opts.Precision.runningStatisticsAlwaysFloat(False)
    deep_copy = copy.deepcopy(opts)

    opts.deviceIterations(4)
    opts.Distributed.configureProcessId(2, 15)
    opts.anchorTensor("t2", "tensor2", OutputMode.Final)
    opts._Popart.set("autoRecomputation", 2)
    opts.TensorLocations.setWeightLocation(locationOutsideChip)
    opts.Precision.halfFloatCasting(
        HalfFloatCastingBehavior.FloatDowncastToHalf)

    assert opts.device_iterations != deep_copy.device_iterations
    assert opts.anchored_tensors != deep_copy.anchored_tensors
    assert opts.replication_factor == deep_copy.replication_factor
    assert opts.log_dir == deep_copy.log_dir
    assert opts.auto_round_num_ipus == deep_copy.auto_round_num_ipus
    assert opts.output_mode == deep_copy.output_mode
    assert opts.output_return_period == deep_copy.output_return_period
    assert opts.connection_type == deep_copy.connection_type
    assert opts.sync_pattern == deep_copy.sync_pattern
    assert (opts.available_memory_proportion ==
            deep_copy.available_memory_proportion)

    assert (opts.Precision.half_float_casting !=
            deep_copy.Precision.half_float_casting)
    assert (opts.Precision.running_statistics_always_float ==
            deep_copy.Precision.running_statistics_always_float)
    assert (opts.Precision.autocast_enabled ==
            deep_copy.Precision.autocast_enabled)

    assert (opts.Distributed.distributed_process_id !=
            deep_copy.Distributed.distributed_process_id)
    assert (opts.Distributed.num_distributed_processes ==
            deep_copy.Distributed.num_distributed_processes)

    assert deep_copy.TensorLocations.location_weight["onChip"]
    assert not opts.TensorLocations.location_weight["onChip"]

    assert (opts._Popart.options["autoRecomputation"] !=
            deep_copy._Popart.options["autoRecomputation"])
    assert (opts._Popart.options["dummyKey"] ==
            deep_copy._Popart.options["dummyKey"])

    assert (opts.Training.gradient_accumulation ==
            deep_copy.Training.gradient_accumulation)

    assert opts.Jit.trace_model == deep_copy.Jit.trace_model


def test_preserving_options_intact():
    class ExampleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            return torch.cat([
                100 * torch.nn.LeakyReLU()(-x + self.bias),
                100 * torch.nn.LeakyReLU()(x - self.bias)
            ],
                             dim=-1)

    class ExampleModelWithLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = ExampleModel()

        def forward(self, input, target):
            out = self.model(input)
            return (torch.nn.functional.softmax(out),
                    torch.nn.CrossEntropyLoss(reduction="mean")(out, target))

    model = ExampleModelWithLoss()
    opts = poptorch.Options()
    training = poptorch.trainingModel(model, opts)
    inference = poptorch.inferenceModel(model, opts)

    assert opts.defaultOutputMode()
    assert training.options.output_mode == OutputMode.Final
    assert inference.options.output_mode == OutputMode.All


@pytest.mark.parametrize("namescopes_enabled", [True, False])
def test_name_scope_hook_disabled(namescopes_enabled):
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 5),
                                              torch.nn.MaxPool2d(2),
                                              torch.nn.ReLU())
            self.layer2 = torch.nn.Sequential(torch.nn.Linear(40, 10),
                                              torch.nn.ReLU())
            self.softmax = torch.nn.LogSoftmax(1)

        def forward(self, x):
            x = self.layer1(x)
            x = x.view(5, 40)
            x = self.layer2(x)
            x = self.softmax(x)
            return x

    model = Network()
    options = poptorch.Options()
    if not namescopes_enabled:
        options.disableModuleNamescope()
    poptorch_model = poptorch.inferenceModel(model, options)

    input = torch.randn(2, 1, 15, 15)
    _ = poptorch_model(input)

    ir = poptorch_model._debugGetPopartIR()  # pylint: disable=protected-access

    expected_namescopes = [
        'layer1/0/', 'layer1/1/', 'layer1/1/', 'layer2/0/', 'layer2/1/',
        'softmax'
    ]
    base_names = ['Conv', 'MaxPool', 'Relu', 'MatMul', 'Relu', 'LogSoftmax']
    assert len(expected_namescopes) == len(base_names)

    for i, name in enumerate(base_names):
        namescope = expected_namescopes[i] if namescopes_enabled else ''
        expected_output = f'"name":"{namescope}{name}'
        assert ir.find(expected_output)


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
        poptorch_model.options.Training,
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


# pylint: disable=protected-access
def test_options_change_after_use():
    model = torch.nn.Linear(10, 10)
    opts = poptorch.Options()
    poptorch_model = helpers.trainingModelWithLoss(
        model, options=opts, loss=torch.nn.CrossEntropyLoss())

    with pytest.raises(Exception):
        opts.randomSeed(42)
    with pytest.raises(Exception):
        poptorch_model.options.set(random_seed=42)
    with pytest.raises(Exception):
        opts.Training.gradientAccumulation(0)
    with pytest.raises(Exception):
        popart_opts = opts._Popart
        opts._Popart.set("groupNormStridedChannelGrouping", True)

    opts = poptorch.Options()
    features = torch.randn([100, 1, 128, 128])
    labels = torch.empty([100], dtype=torch.long).random_(10)
    dataset = torch.utils.data.TensorDataset(features, labels)

    poptorch_data_loader = poptorch.DataLoader(
        opts,
        dataset=dataset,
    )
    with pytest.raises(Exception):
        opts.randomSeed(42)
    with pytest.raises(Exception):
        poptorch_data_loader.options.set(random_seed=42)
    with pytest.raises(Exception):
        poptorch_data_loader.Training.gradientAccumulation(0)
    with pytest.raises(Exception):
        popart_opts = poptorch_data_loader._Popart
        popart_opts.set("groupNormStridedChannelGrouping", True)


def test_wrap_options():
    """Popdist wraps poptorch Options using something similar"""

    class _Distributed(poptorch.options._DistributedOptions):
        pass

    opts = poptorch.Options()
    opts.Distributed.__class__ = _Distributed
