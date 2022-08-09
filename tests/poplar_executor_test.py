#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import datetime
import unittest.mock
import os
import re
import tempfile
import glob
import warnings

import pytest
import torch
import torch.multiprocessing as mp
import helpers
import poptorch


@pytest.mark.ipuHardwareRequired
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.parametrize("trace_model", [True, False])
def test_ExecutableCaching(capfd, trace_model):
    class Model(torch.nn.Module):
        def forward(self, x):
            return x * 6

    with tempfile.TemporaryDirectory() as cache:
        opts = poptorch.Options()
        opts.enableExecutableCaching(cache)
        opts.Jit.traceModel(trace_model)
        m = poptorch.inferenceModel(Model(), opts)
        m.compile(torch.rand(2, 3))
        m.destroy()
        log = helpers.LogChecker(capfd)
        log.assert_contains("set enableEngineCaching to value true")
        assert len(os.listdir(cache)) == 1, "No executable saved in the cache"

        n = poptorch.inferenceModel(Model(), opts)
        n.compile(torch.rand(2, 3))
        log = helpers.LogChecker(capfd)
        log.assert_contains("set enableEngineCaching to value true")


@pytest.mark.ipuHardwareRequired
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.parametrize("trace_model", [True, False])
def test_ExecutableCaching_env(capfd, trace_model):
    class Model(torch.nn.Module):
        def forward(self, x):
            return x * 6

    with tempfile.TemporaryDirectory() as cache:
        os.environ["POPTORCH_CACHE_DIR"] = cache
        opts = poptorch.Options()
        opts.Jit.traceModel(trace_model)
        m = poptorch.inferenceModel(Model(), opts)
        m.compile(torch.rand(2, 3))
        m.destroy()
        log = helpers.LogChecker(capfd)
        log.assert_contains("set enableEngineCaching to value true")
        assert len(os.listdir(cache)) == 1, "No executable saved in the cache"

        n = poptorch.inferenceModel(Model(), opts)
        n.compile(torch.rand(2, 3))
        log = helpers.LogChecker(capfd)
        log.assert_contains("set enableEngineCaching to value true")


class Network(torch.nn.Module):
    def forward(self, x, y):
        return x + y


def _create_model_and_export(opts, filename):
    model = Network()

    inference_model = poptorch.inferenceModel(model, opts)
    x = torch.ones(2)
    y = torch.zeros(2)

    inference_model.compileAndExport(filename, x, y)
    assert os.path.exists(filename)


@unittest.mock.patch.dict("os.environ", helpers.disableAllModels())
@pytest.mark.parametrize("trace_model", [True, False])
def test_offline_ipu_compileAndExport_file(trace_model, filename=None):
    # Force-disable the IPU model
    opts = poptorch.Options().useOfflineIpuTarget()
    opts.Jit.traceModel(trace_model)

    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, "model.poptorch")
        _create_model_and_export(opts, filename)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_precompile_then_load(trace_model):
    opts = poptorch.Options().useOfflineIpuTarget(
        poptorch.ipuHardwareVersion())
    opts.Jit.traceModel(trace_model)
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, "model.poptorch")
        _create_model_and_export(opts, filename)

        poptorch_model = poptorch.load(filename)

        x = torch.tensor([1., 2.])
        y = torch.tensor([3., 4.])
        # Check the user model was restored
        helpers.assert_allclose(actual=poptorch_model.model(x, y),
                                expected=torch.tensor([4., 6.]))
        helpers.assert_allclose(actual=poptorch_model(x, y),
                                expected=torch.tensor([4., 6.]))


@unittest.mock.patch.dict("os.environ", helpers.disableAllModels())
@pytest.mark.parametrize("trace_model", [True, False])
def test_offline_ipu_compileAndExport_dir(trace_model):
    class Network(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = Network()
    # Force-disable the IPU model
    opts = poptorch.Options().useOfflineIpuTarget()
    opts.Jit.traceModel(trace_model)
    poptorch.inferenceModel(model, opts)

    inference_model = poptorch.inferenceModel(model, opts)
    x = torch.ones(2)
    y = torch.zeros(2)

    with tempfile.TemporaryDirectory() as tmp:
        assert os.path.isdir(tmp)
        # Model is local to the function: it cannot be serialised so don't
        # export it.
        inference_model.compileAndExport(tmp, x, y, export_model=False)
        files = glob.glob(f"{tmp}/*")
        assert len(files) == 1, "Expected exactly 1 file"


@pytest.mark.parametrize("trace_model", [True, False])
def test_inference_attributes(trace_model):
    class Model(torch.nn.Module):
        def __init__(self, attr):
            super().__init__()
            self.attr = attr

        def getAttr(self):
            return self.attr

        def forward(self, x, y):
            return x + y + 5

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(Model("MyAttr"), options)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])

    poptorch_model(t1, t2)

    assert poptorch_model.getAttr() == poptorch_model.attr
    assert poptorch_model.attr == "MyAttr"


@pytest.mark.parametrize("trace_model", [True, False])
def test_training_attributes(trace_model):
    def custom_loss(output, target):
        # Mean squared error with a scale
        loss = output - target
        loss = loss * loss * 5
        return poptorch.identity_loss(loss, reduction="mean")

    class Model(torch.nn.Module):
        def __init__(self, attr):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))
            self.attr = attr

        def getAttr(self):
            return self.attr

        def forward(self, x, target):
            x = x + 1
            x = poptorch.ipu_print_tensor(x) + self.bias
            return x, custom_loss(x, target)

    model = Model("MyAttr")
    input = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([30.0, 40.0, 50.0])
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)

    poptorch_model(input, target)

    assert poptorch_model.getAttr() == poptorch_model.attr
    assert poptorch_model.attr == "MyAttr"


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("use_half", [False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_explicit_destroy(use_half, trace_model):
    class ExampleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            x = x + 1

            # It is important to make sure the result of the print is used.
            x = poptorch.ipu_print_tensor(x)

            return x + self.bias

    def custom_loss(output, target):
        # Mean squared error with a scale
        loss = output - target
        loss = loss * loss * 5
        return poptorch.identity_loss(loss, reduction="mean")

    class ExampleModelWithCustomLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = ExampleModel()

        def forward(self, input, target=None):
            out = self.model(input)
            if target is not None:
                return out, custom_loss(out, target)
            return out

    opts = poptorch.Options()
    opts.Jit.traceModel(trace_model)
    # Both models will use the same IPU device.
    opts.useIpuId(1)

    model = ExampleModelWithCustomLoss()
    input = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([30.0, 40.0, 50.0])
    if use_half:
        model.half()
        input = input.half()
        target = target.half()
    training_model = poptorch.trainingModel(model, opts)
    inference_model = poptorch.inferenceModel(model, opts)

    training_model(input=input, target=target)
    training_model.destroy()

    error_msg = r"Model has not been compiled or has been destroyed."
    with pytest.raises(poptorch.Error, match=error_msg):
        training_model.copyWeightsToHost()
    with pytest.raises(poptorch.Error, match=error_msg):
        training_model.copyWeightsToDevice()

    inference_model(input)


def _compile_model_offline(trace_model, cache, pid, num_processes):
    poptorch.setLogLevel("DEBUG")  # Force debug logging in worker process
    opts = poptorch.Options().useOfflineIpuTarget()
    opts.enableExecutableCaching(cache)
    # Disable compilation bar to avoid issues with capfd
    opts.showCompilationProgressBar(False)
    opts.deviceIterations(10)
    opts.Distributed.configureProcessId(pid, num_processes)
    opts.Jit.traceModel(trace_model)

    class ModelWithLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, data, target):
            out = self.linear(data)
            loss = self.loss(out, target)
            return out, loss

    model = ModelWithLoss()
    poptorch_model = poptorch.trainingModel(model, options=opts)

    # 10 Batches of 10.
    input = torch.randn(10, 10)
    # 10 batches of 1
    label = torch.randint(0, 10, [1])
    label = label.expand([10])

    poptorch_model.compile(input, label)


# Force-disable the IPU model
@unittest.mock.patch.dict("os.environ", helpers.disableAllModels())
@helpers.printCapfdOnExit
@pytest.mark.parametrize("trace_model", [True, False])
def test_distributed_compile(capfd, trace_model):
    num_processes = 6
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "poptorch_cache")

        ctx = mp.get_context('spawn')
        processes = [
            ctx.Process(target=_compile_model_offline,
                        args=(trace_model, cache, pid, num_processes))
            for pid in range(num_processes)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def getTimestamp(line):
        m = re.match(r"\[([\d:.]+)\]", line)
        return datetime.datetime.strptime(m.group(1), "%H:%M:%S.%f")

    log = helpers.LogChecker(capfd).createIterator()
    includes_compilation = True
    for p in processes:
        start = getTimestamp(log.findNext("cache file locked"))
        end = getTimestamp(log.findNext("released the cache lock"))

        if includes_compilation:
            assert end - start > datetime.timedelta(seconds=1), (
                "Expected the"
                " first process model compilation to take more than 1 "
                f"second but it took {end - start}")
        else:
            assert end - start < datetime.timedelta(seconds=1), (
                "Expected "
                "processes to load the executable from the cache in under"
                f" 1 second but it took {end - start}")
        includes_compilation = False


def test_nondeterministic_warning_filter():
    # This simple model generates a few jit warnings including the
    # non-deterministic ones that we filter in poptorch.  This test checks that
    # these additional warnings are still emitted.
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.uniform = torch.distributions.Uniform(-1.0, 1.0)

        def forward(self):
            sz = torch.tensor((20, ))
            return self.uniform.sample(sz)

    model = Model()

    # trace and capture all warnings for a baseline
    with warnings.catch_warnings(record=True) as native_warnings:
        torch.jit.trace(model, ())

    jit_warns = set(str(w.message) for w in native_warnings)

    options = poptorch.Options()
    options.Jit.traceModel(True)
    # compile with poptorch and capture all warnings
    with warnings.catch_warnings(record=True) as filtered_warnings:
        poptorch.inferenceModel(model, options).compile()

    pop_warns = set(str(w.message) for w in filtered_warnings)

    # The only differences in warnings should be the filtered ones
    remainder = list(jit_warns - pop_warns)
    assert len(remainder) == 2, "Expected exactly two filtered warnings"

    expected_filtered_warnings = [
        "Trace had nondeterministic nodes",
        "the traced function does not match the corresponding output"
    ]

    for r in remainder:
        assert any([
            f in r for f in expected_filtered_warnings
        ]), f"Compilation generated unexpected warning.\nActual warning: {r}"


@pytest.mark.mlirSupportRequired
def test_dispatcher_cpu_output():
    const1 = torch.tensor([1, 2])
    const2 = torch.tensor([3, 4])

    class Model(torch.nn.Module):
        def forward(self):
            return (const1 + const2, ([const1, const2], [const1,
                                                         const2]), const2)

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(False)

    with warnings.catch_warnings(record=True) as filtered_warnings:
        poptorch.inferenceModel(model, options).compile()

    pop_warns = set(str(w.message) for w in filtered_warnings)

    expected_warning = "Output expected to be on the IPU but is on cpu"

    for r in pop_warns:
        assert expected_warning in r, (f"Compilation generated unexpected "
                                       f"warning.\nActual warning: {r}")


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_get_cycles_error_msgs(trace_model):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(Model(), options)

    error_msg = (r"Cycle count logging is disabled. Please set option "
                 r"logCycleCount to True to enable.")
    with pytest.raises(poptorch.Error, match=error_msg):
        inference_model.cycleCount()

    opts = poptorch.Options()
    opts.logCycleCount(True)
    opts.Jit.traceModel(trace_model)

    inference_model = poptorch.inferenceModel(Model(), options=opts)

    error_msg = (r"Please run the model at least once before obtaining cycle "
                 r"count.")
    with pytest.raises(poptorch.Error, match=error_msg):
        inference_model.cycleCount()

    inference_model.compile(torch.Tensor([1.0]), torch.Tensor([2.0]))

    error_msg = (r"Please run the model at least once before obtaining cycle "
                 r"count.")
    with pytest.raises(poptorch.Error, match=error_msg):
        inference_model.cycleCount()

    inference_model(torch.Tensor([3.0]), torch.Tensor([4.0]))
    assert inference_model.cycleCount() > 0


@pytest.mark.skipif(poptorch.ipuHardwareIsAvailable(),
                    reason="Test error message when no hardware")
@pytest.mark.parametrize("trace_model", [True, False])
def test_get_cycles_no_hw(trace_model):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(Model(), options)

    opts = poptorch.Options()
    opts.logCycleCount(True)
    opts.Jit.traceModel(trace_model)

    inference_model = poptorch.inferenceModel(Model(), options=opts)

    error_msg = (
        r"Cycle count logging is only supported on actual IPU hardware.")
    with pytest.raises(poptorch.Error, match=error_msg):
        inference_model(torch.Tensor([3.0]), torch.Tensor([4.0]))


@pytest.mark.parametrize("rewrap_executor", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_rewrap_model(rewrap_executor, trace_model):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 1)
            self.loss = torch.nn.L1Loss()

        def forward(self, x):
            y = self.fc(x)
            loss = self.loss(y, x + 1)

            return loss

    model = Model()

    opts = poptorch.Options()
    opts.Jit.traceModel(trace_model)

    # Normal running
    torch.nn.init.ones_(model.fc.weight)
    torch.nn.init.zeros_(model.fc.bias)

    opts.deviceIterations(10)
    poptorch_model = poptorch.trainingModel(model, options=opts)

    poptorch_model(torch.ones([10]))

    bias_after_1000 = float(model.fc.bias)

    # Try rewrapping model half way
    torch.nn.init.ones_(model.fc.weight)
    torch.nn.init.zeros_(model.fc.bias)

    with pytest.raises(AssertionError):
        helpers.assert_allclose(actual=model.fc.bias, expected=bias_after_1000)

    model.destroy()

    opts = poptorch.Options()
    opts.deviceIterations(5)
    poptorch_model = poptorch.trainingModel(model, options=opts)

    poptorch_model(torch.ones([5]))

    err_msg = (r"Model has already been wrapped in 'poptorch.trainingModel'."
               r" Call model.destroy\(\) on the model to unwrap before "
               "wrapping again.")
    with pytest.raises(RuntimeError, match=err_msg):
        poptorch_model = poptorch.trainingModel(model, options=opts)

    # re-wrap for test
    if rewrap_executor:
        poptorch_model.destroy()
        poptorch_model = poptorch.trainingModel(poptorch_model, options=opts)
    else:
        model.destroy()
        poptorch_model = poptorch.trainingModel(model, options=opts)

    poptorch_model(torch.ones([5]))
    helpers.assert_allclose(actual=float(model.fc.bias),
                            expected=bias_after_1000)
