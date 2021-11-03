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


def _compileAndExport(filename, export_model=True, training=True):
    torch.manual_seed(42)
    model = ExampleModelWithLoss()

    input = torch.randn(1, 10)
    target = torch.randint(0, 10, [1])

    opts = poptorch.Options()
    opts.useOfflineIpuTarget(poptorch.ipuHardwareVersion())

    if training:
        model.train()
        poptorch_model = poptorch.trainingModel(model, opts)
        poptorch_model.compileAndExport(filename,
                                        input,
                                        target,
                                        export_model=export_model)
    else:
        model.eval()
        poptorch_model = poptorch.inferenceModel(model, opts)
        poptorch_model.compileAndExport(filename,
                                        input,
                                        export_model=export_model)
    poptorch_model.destroy()
    return input, target


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_export_then_load():
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, "model.poptorch")
        input, target = _compileAndExport(filename)

        poptorch_model = poptorch.load(filename)
        poptorch_model(input, target)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_export_then_load_setIpu():
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, "model.poptorch")
        input, target = _compileAndExport(filename)

        def setIpuDevice(opts):
            opts.useIpuId(1)  # always use IPU 1

        poptorch_model = poptorch.load(filename, edit_opts_fn=setIpuDevice)
        poptorch_model(input, target)


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
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


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_export_train_validate_no_python():
    with tempfile.TemporaryDirectory() as tmp:
        train_filename = os.path.join(tmp, "train.poptorch")
        valid_filename = os.path.join(tmp, "valid.poptorch")
        input, target = _compileAndExport(train_filename, export_model=False)
        _compileAndExport(valid_filename, export_model=False, training=False)

        model = ExampleModelWithLoss()
        training_model = poptorch.trainingModel(model)
        training_model.loadExecutable(train_filename)

        model.eval()
        validation_model = poptorch.inferenceModel(model)
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


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_export_train_validate():
    with tempfile.TemporaryDirectory() as tmp:
        train_filename = os.path.join(tmp, "train.poptorch")
        valid_filename = os.path.join(tmp, "valid.poptorch")
        input, target = _compileAndExport(train_filename)
        _compileAndExport(valid_filename, training=False)

        training_model = poptorch.load(train_filename)
        validation_model = poptorch.inferenceModel(training_model)
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


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_export_train_save_validate():
    with tempfile.TemporaryDirectory() as tmp:
        train_filename = os.path.join(tmp, "train.poptorch")
        valid_filename = os.path.join(tmp, "valid.poptorch")
        input, target = _compileAndExport(train_filename)

        training_model = poptorch.load(train_filename)
        opts = poptorch.Options()
        opts.useOfflineIpuTarget(poptorch.ipuHardwareVersion())
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


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
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


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
def test_export_train_save_validate_load_weights():
    with tempfile.TemporaryDirectory() as tmp:
        train_filename = os.path.join(tmp, "train.poptorch")
        valid_filename = os.path.join(tmp, "valid.poptorch")
        weights_filename = os.path.join(tmp, "weights.poptorch")
        _compileAndExport(valid_filename, training=False)
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
