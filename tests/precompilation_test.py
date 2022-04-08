#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import tempfile

import pytest
import torch
import poptorch


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
# TODO(T51159) Support dispatch tracing + serialized executables
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
def test_export_then_load():
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, "model.poptorch")
        input, target = _compileAndExport(filename)

        poptorch_model = poptorch.load(filename)
        poptorch_model(input, target)


@pytest.mark.ipuHardwareRequired
def test_export_then_load_setIpu():
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, "model.poptorch")
        input, target = _compileAndExport(filename)

        def setIpuDevice(opts):
            opts.useIpuId(1)  # always use IPU 1

        poptorch_model = poptorch.load(filename, edit_opts_fn=setIpuDevice)
        poptorch_model(input, target)


@pytest.mark.ipuHardwareRequired
def test_export_no_python_then_load():
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, "model.poptorch")
        input, target = _compileAndExport(filename, export_model=False)

        # load_exe_start
        model = ExampleModelWithLoss()

        opts = poptorch.Options()
        poptorch_model = poptorch.trainingModel(model, opts)
        poptorch_model.loadExecutable(filename)

        poptorch_model(input, target)


@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_export_train_validate_no_python(trace_model):
    with tempfile.TemporaryDirectory() as tmp:
        train_filename = os.path.join(tmp, "train.poptorch")
        valid_filename = os.path.join(tmp, "valid.poptorch")
        input, target = _compileAndExport(train_filename, export_model=False)
        _compileAndExport(valid_filename,
                          export_model=False,
                          training=False,
                          trace_model=trace_model)

        model = ExampleModelWithLoss()
        training_model = poptorch.trainingModel(model)
        training_model.loadExecutable(train_filename)

        model.eval()
        options = poptorch.Options()
        options.Jit.traceModel(trace_model)
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
        input, target = _compileAndExport(train_filename)
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
        input, target = _compileAndExport(train_filename)

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
def test_export_train_save_train():
    with tempfile.TemporaryDirectory() as tmp:
        train_filename = os.path.join(tmp, "train.poptorch")
        weights_filename = os.path.join(tmp, "weights.poptorch")
        input, target = _compileAndExport(train_filename)

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
        input, target = _compileAndExport(train_filename)

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
