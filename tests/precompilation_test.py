#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import re
import marshal
import subprocess
import sys
import json
import pathlib
import tempfile
import unittest.mock

import pytest
import torch
import poptorch
import helpers
if helpers.is_running_tests:
    import pva


class ExampleModelWithLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, target=None):
        fc = self.fc(x)
        if self.training:
            return fc, self.loss(fc, target)
        return fc


def _createExampleModel(training, trace_model, offline_target=False):
    torch.manual_seed(42)
    model = ExampleModelWithLoss()

    opts = poptorch.Options()
    if offline_target:
        opts.useOfflineIpuTarget(poptorch.ipuHardwareVersion())
    opts.Jit.traceModel(trace_model)

    if training:
        model.train()
        poptorch_model = poptorch.trainingModel(model, opts)
    else:
        model.eval()
        poptorch_model = poptorch.inferenceModel(model, opts)
    return poptorch_model


def _compileAndExport(filename,
                      export_model=True,
                      training=True,
                      trace_model=True):
    poptorch_model = _createExampleModel(training, trace_model, True)

    input = torch.randn(1, 10)
    target = torch.randint(0, 10, [1])

    if training:
        poptorch_model.compileAndExport(filename,
                                        input,
                                        target,
                                        export_model=export_model)
    else:
        poptorch_model.compileAndExport(filename,
                                        input,
                                        export_model=export_model)
    poptorch_model.destroy()
    return input, target


@pytest.mark.ipuHardwareRequired
# TODO(T64293) Support dispatch tracing + serialized executables
#@pytest.mark.parametrize("trace_model", [True, False])
def test_export_then_load_live_model(trace_model=True):
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, "model.poptorch")
        poptorch_model = _createExampleModel(training=False,
                                             trace_model=trace_model)

        input = torch.randn(1, 10)
        # Running the model will trigger the executable compilation
        poptorch_model(input)
        # Save the executable and destroy the model
        poptorch_model.save(filename)
        poptorch_model.destroy()

        # Reload the model from file and run it.
        poptorch_model = poptorch.load(filename)
        poptorch_model(input)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_export_then_load(trace_model):
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, "model.poptorch")
        input, target = _compileAndExport(filename, trace_model=trace_model)

        poptorch_model = poptorch.load(filename)
        poptorch_model(input, target)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_export_then_load_setIpu(trace_model):
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, "model.poptorch")
        input, target = _compileAndExport(filename, trace_model=trace_model)

        def setIpuDevice(opts):
            opts.useIpuId(1)  # always use IPU 1

        poptorch_model = poptorch.load(filename, edit_opts_fn=setIpuDevice)
        poptorch_model(input, target)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_export_no_python_then_load(trace_model):
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, "model.poptorch")
        input, target = _compileAndExport(filename,
                                          export_model=False,
                                          trace_model=trace_model)

        # load_exe_start
        model = ExampleModelWithLoss()

        opts = poptorch.Options()
        opts.Jit.traceModel(trace_model)
        poptorch_model = poptorch.trainingModel(model, opts)
        poptorch_model.loadExecutable(filename)

        poptorch_model(input, target)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_export_train_validate_no_python(trace_model):
    with tempfile.TemporaryDirectory() as tmp:
        train_filename = os.path.join(tmp, "train.poptorch")
        valid_filename = os.path.join(tmp, "valid.poptorch")
        input, target = _compileAndExport(train_filename,
                                          export_model=False,
                                          trace_model=trace_model)
        _compileAndExport(valid_filename,
                          export_model=False,
                          training=False,
                          trace_model=trace_model)

        model = ExampleModelWithLoss()
        options = poptorch.Options()
        options.Jit.traceModel(trace_model)
        training_model = poptorch.trainingModel(model, options=options)
        training_model.loadExecutable(train_filename)

        model.eval()
        validation_model = poptorch.inferenceModel(model, options)
        validation_model.loadExecutable(valid_filename)

        # Make sure the first run doesn't already pass the test.
        out, original_loss = training_model(input, target)
        assert torch.argmax(out, dim=1) != target

        out = validation_model(input)
        assert torch.argmax(out, dim=1) != target

        for _ in range(500):
            out, loss = training_model(input, target)

        # Check we have trained the model
        assert loss < original_loss
        assert loss < 0.05
        assert torch.argmax(out, dim=1) == target

        # Check validation model has the weights
        out = validation_model(input)
        assert torch.argmax(out, dim=1) == target


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_export_train_validate(trace_model):
    with tempfile.TemporaryDirectory() as tmp:
        train_filename = os.path.join(tmp, "train.poptorch")
        valid_filename = os.path.join(tmp, "valid.poptorch")
        input, target = _compileAndExport(train_filename,
                                          trace_model=trace_model)
        _compileAndExport(valid_filename,
                          training=False,
                          trace_model=trace_model)

        training_model = poptorch.load(train_filename)
        options = poptorch.Options()
        options.Jit.traceModel(trace_model)
        validation_model = poptorch.inferenceModel(training_model, options)
        validation_model.model.eval()
        validation_model.loadExecutable(valid_filename)

        # Make sure the first run doesn't already pass the test.
        out, original_loss = training_model(input, target)
        assert torch.argmax(out, dim=1) != target

        out = validation_model(input)
        assert torch.argmax(out, dim=1) != target

        for _ in range(500):
            out, loss = training_model(input, target)

        # Check we have trained the model
        assert loss < original_loss
        assert loss < 0.05
        assert torch.argmax(out, dim=1) == target

        # Check validation model has the weights
        out = validation_model(input)
        assert torch.argmax(out, dim=1) == target


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_export_train_save_validate(trace_model):
    with tempfile.TemporaryDirectory() as tmp:
        train_filename = os.path.join(tmp, "train.poptorch")
        valid_filename = os.path.join(tmp, "valid.poptorch")
        input, target = _compileAndExport(train_filename,
                                          trace_model=trace_model)

        training_model = poptorch.load(train_filename)
        opts = poptorch.Options()
        opts.useOfflineIpuTarget(poptorch.ipuHardwareVersion())
        opts.Jit.traceModel(trace_model)
        validation_model = poptorch.inferenceModel(training_model, opts)
        validation_model.model.eval()

        # Make sure the first run doesn't already pass the test.
        out, original_loss = training_model(input, target)
        assert torch.argmax(out, dim=1) != target

        # Now train the model
        for _ in range(500):
            out, loss = training_model(input, target)

        # Check we have trained the model
        assert loss < original_loss
        assert loss < 0.05
        assert torch.argmax(out, dim=1) == target

        validation_model.compileAndExport(valid_filename, input)
        validation_model = poptorch.load(valid_filename)

        # Check validation model has the weights
        out = validation_model(input)
        assert torch.argmax(out, dim=1) == target


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_export_train_save_train(trace_model):
    with tempfile.TemporaryDirectory() as tmp:
        train_filename = os.path.join(tmp, "train.poptorch")
        weights_filename = os.path.join(tmp, "weights.poptorch")
        input, target = _compileAndExport(train_filename,
                                          trace_model=trace_model)

        training_model = poptorch.load(train_filename)

        # Make sure the first run doesn't already pass the test.
        out, original_loss = training_model(input, target)
        assert torch.argmax(out, dim=1) != target

        # Now train the model
        for _ in range(500):
            out, loss = training_model(input, target)

        # Check we have trained the model
        assert loss < original_loss
        assert loss < 0.05
        assert torch.argmax(out, dim=1) == target

        torch.save(training_model.model.state_dict(), weights_filename)
        training_model.destroy()

        training_model = poptorch.load(train_filename)
        training_model.load_state_dict(torch.load(weights_filename))

        # Check we still have the trained weights
        out, loss = training_model(input, target)
        assert loss < original_loss
        assert loss < 0.05
        assert torch.argmax(out, dim=1) == target


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_export_train_save_validate_load_weights(trace_model):
    with tempfile.TemporaryDirectory() as tmp:
        train_filename = os.path.join(tmp, "train.poptorch")
        valid_filename = os.path.join(tmp, "valid.poptorch")
        weights_filename = os.path.join(tmp, "weights.poptorch")
        _compileAndExport(valid_filename,
                          training=False,
                          trace_model=trace_model)
        input, target = _compileAndExport(train_filename,
                                          trace_model=trace_model)

        training_model = poptorch.load(train_filename)

        # Make sure the first run doesn't already pass the test.
        out, original_loss = training_model(input, target)
        assert torch.argmax(out, dim=1) != target

        # Now train the model
        for _ in range(500):
            out, loss = training_model(input, target)

        # Check we have trained the model
        assert loss < original_loss
        assert loss < 0.05
        assert torch.argmax(out, dim=1) == target

        torch.save(training_model.model, weights_filename)
        training_model.destroy()

        validation_model = poptorch.load(valid_filename)
        validation_model.load_state_dict(
            torch.load(weights_filename).state_dict())

        # Check validation model has the weights
        out = validation_model(input)
        assert torch.argmax(out, dim=1) == target


def process_to_generate_profiling_data():
    """A function executed as a script running in a separate process.
    We need to do this because profiling data is only written to disk
    when a process exits.
    """
    # pylint: disable=import-outside-toplevel
    # pylint: disable=reimported
    import poptorch
    import torch

    class Block(torch.nn.Module):
        def __init__(self, num_hidden):
            super().__init__()
            self.softmax = torch.nn.LogSoftmax(1)
            self.lstm = torch.nn.LSTM(3, num_hidden)

        def forward(self, x):
            x, _ = self.lstm(x)
            x = self.softmax(x)
            return x

    class Model(torch.nn.Module):
        def __init__(self, num_hidden):
            super().__init__()
            self.relu = torch.nn.ReLU()
            self.block0 = Block(num_hidden)

        def forward(self, x):
            x = self.block0(x)
            x = self.relu(x)
            loss = poptorch.identity_loss(x**2, reduction='sum')
            return x, loss

    input = torch.randn(1, 1, 3)
    model = Model(3)

    optimizer = poptorch.optim.SGD(model.parameters(), lr=0.0)

    opts = poptorch.Options()
    opts.useOfflineIpuTarget()
    # Profiling information is incomplete when using JIT.
    opts.Jit.traceModel(False)
    training_model = poptorch.trainingModel(model, opts, optimizer=optimizer)

    training_model.compile(input)


@unittest.mock.patch.dict(
    "os.environ", {
        **helpers.disableAllModels(), "POPLAR_ENGINE_OPTIONS":
        json.dumps({
            "autoReport.directory": ".",
            "autoReport.all": "true",
            "autoReport.outputDebugInfo": "true",
            "autoReport.outputExecutionProfile": "false"
        })
    })
@pytest.mark.mlirSupportRequired
def test_pva_annotations():
    def findPoptorchLayer(op):
        layer = json.loads(op.layer)["layer"]
        if layer == "poptorch":
            return op
        assert op.parents, "Can't find 'poptorch' layer"
        return findPoptorchLayer(op.parents[0])

    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        subprocess.check_output(
            [
                sys.executable,
                "-u",  # needed to ensure messages are sent to stdout immediately
                "-c",
                f"""
import os, marshal, types;code = marshal.loads({marshal.dumps(process_to_generate_profiling_data.__code__)})
fn = types.FunctionType(code, globals(), "generate_profiling_data")
fn()
        """
            ],
            universal_newlines=True,
            env=os.environ)

        debug = pathlib.Path("debug.cbor").resolve(strict=True)
        profile = pathlib.Path("training", "profile.pop").resolve(strict=True)

        # Read this file and find where the layers were called from to make
        # sure the line numbers are correct inside the profiling information.
        it = helpers.LogIterator(open(__file__, "r").read().split("\n"))
        lines = []
        it.findNext(re.escape("def process_to_generate_profiling_data():"))
        for e in [
                "self.lstm(", "self.softmax(", "self.relu(", "identity_loss("
        ]:
            it.findNext(re.escape(e))
            lines.append(it.lineNumber())

        report = pva.openReport(str(profile), str(debug))
        op_analysis = pva.OperationAnalysis(report)
        for op in op_analysis.operations:
            if not op.name or op.name == "Call":
                continue
            if op.replacedDebugContext:
                ctx = op.replacedDebugContext[0]
            else:
                ctx = op.debugContext
            pop_op = findPoptorchLayer(ctx)
            data = json.loads(pop_op.json)
            op_file = pop_op.location.fileName
            op_line = pop_op.location.lineNumber
            print(f"Name {op.name} {op_file}:{op_line} Debug {data}")

            # All the ops should be associated to this file
            assert os.path.realpath(op_file) == os.path.realpath(__file__)

            assert op.name == data["op_name"]

            # The identity loss is not a layer in the model therefore it won't have a prefix.
            if data["op_type"] in ["Pow", "Identityloss"]:
                assert data["op_name"] == data["op_type"]
            else:
                # All the other ops are stored in the model therefore they'll have prefix
                # "foo/op_type" where "foo" is the name of the attribute in the model.
                assert data["op_name"].endswith("/" + data["op_type"])
            assert data["layer"] == "poptorch"

            if data["op_name"].startswith("block0/lstm"):
                assert op_line == lines[0]
            elif data["op_name"].startswith("block0/softmax"):
                assert op_line == lines[1]
            elif data["op_name"].startswith("relu/"):
                assert op_line == lines[2]
            elif data["op_name"] == data["op_type"]:  # identity_loss(x**2)
                assert op_line == lines[3]
            else:
                raise ValueError("Unexpected op " + data["op_name"])
