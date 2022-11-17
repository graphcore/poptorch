#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import re
import math
import unittest.mock
import pytest
import torch
import helpers
import poptorch


@unittest.mock.patch.dict("os.environ", {"POPTORCH_WAIT_FOR_IPU": "0"})
@pytest.mark.ipuHardwareRequired
def test_attach_detach():
    torch.manual_seed(42)

    target = torch.randint(0, 10, [1])
    target = target.expand([10])
    input = torch.randn(10, 10)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, data, target=None):
            out = self.linear(data)

            if target is None:
                return out

            loss = self.loss(out, target)
            return out, loss

    model = Model()

    opts = poptorch.Options()
    # Ensure that both models use the same IPU
    opts.useIpuId(1)
    training = poptorch.trainingModel(model, options=opts)

    opts = opts.clone()
    inference = poptorch.inferenceModel(model, options=opts)

    _, initial_loss = training(input, target)

    if math.isnan(initial_loss):
        raise ValueError("original_loss is NaN")

    if poptorch.ipuHardwareIsAvailable():
        with pytest.raises(poptorch.Error) as excinfo:
            inference.compile(torch.randn(10))
            assert excinfo.match("Failed to acquire")

    training.detachFromDevice()
    # Ensure that this breaks

    error_msg = r"Device is not attached"
    with pytest.raises(poptorch.Error, match=error_msg):
        training.detachFromDevice()

    inference.compile(torch.randn(10))

    if poptorch.ipuHardwareIsAvailable():
        inference.detachFromDevice()

    assert initial_loss > 0.1

    loss = float('nan')

    for _ in range(0, 2):
        _, loss = training(input, target)
        # Each batch should NOT report its own loss. As by default training
        # model should have a "Final" output mode.
        assert len(loss.size()) == 0

        if math.isnan(loss):
            raise ValueError("loss is NaN")

        training.detachFromDevice()

        inference(torch.randn(10))
        inference.detachFromDevice()


@pytest.mark.ipuHardwareRequired
def test_attach_detach_accuracy():
    class TrainingModelWithLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(1, 2)
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, args, loss_inputs=None):
            output = self.model(args)
            if loss_inputs is None:
                return output
            final_loss = self.loss(output, loss_inputs)
            return output, final_loss

    torch.manual_seed(42)

    input_data = torch.Tensor([[1.], [-1.]])
    labels_data = torch.Tensor([0, 1]).long()
    model_with_loss = TrainingModelWithLoss()
    optimizer = poptorch.optim.SGD(model_with_loss.parameters(),
                                   lr=0.1,
                                   use_combined_accum=False)
    training_model = poptorch.trainingModel(model_with_loss,
                                            optimizer=optimizer)
    inference_model = poptorch.inferenceModel(model_with_loss)

    losses1 = []
    for _ in range(5):
        _, loss = training_model(input_data, labels_data)
        print("Loss:", loss)
        losses1.append(loss)

    training_model.detachFromDevice()
    inference1 = inference_model(input_data)
    print("Predictions:", inference1)
    inference_model.detachFromDevice()

    losses2 = []
    for _ in range(100):
        _, loss = training_model(input_data, labels_data)
        print(loss)
        losses2.append(loss)

    training_model.detachFromDevice()
    inference2 = inference_model(input_data)
    print("Predictions:", inference2)

    assert not torch.allclose(inference1, inference2)
    assert not torch.allclose(inference2, torch.zeros(2, 2))
    assert losses1[-1] > losses2[-1]
    for i in range(len(losses2) - 1):
        assert losses2[i] != losses2[i + 1]
    assert losses2[-1] < 0.1


@pytest.mark.ipuHardwareRequired
@unittest.mock.patch.dict("os.environ", {"POPTORCH_WAIT_FOR_IPU": "0"})
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_on_demand_attach(capfd):
    model = torch.nn.Linear(1, 2)

    opts = poptorch.Options()
    opts.connectionType(poptorch.ConnectionType.OnDemand)

    m = poptorch.inferenceModel(model, opts)

    input = torch.Tensor([[1.], [-1.]])
    m(input)
    log = helpers.LogChecker(capfd).createIterator()
    # We acquire device 0 to compile. (It's the first device with a matching target)
    log.findNext(re.escape("Acquired 1 IPU(s): running on device Id 0"))
    # Make sure we compile before we attach to the device.
    log.findNext("Finished Poplar compilation")
    # Device 0 is still free so we'll attach to it.
    log.findNext("Attached to device 0")

    n = poptorch.inferenceModel(model, opts)
    n(input)
    log = helpers.LogChecker(capfd).createIterator()
    # We acquire device 0 to compile. (It's the first device with a matching target)
    # Note: acquiring doesn't mean attaching, it's ok if the device is not actually free.
    log.findNext(re.escape("Acquired 1 IPU(s): running on device Id 0"))
    # Make sure we compile before we attach to the device.
    log.findNext("Finished Poplar compilation")
    # Device 0 is in use by model 'm' so we should automatically get device 1.
    log.findNext("Attached to device 1")

    opts_always = opts.clone()
    opts_always.connectionType(poptorch.ConnectionType.Always)
    o = poptorch.inferenceModel(model, opts_always)
    o(input)
    log = helpers.LogChecker(capfd).createIterator()
    # In Always mode we find a free IPU before the compilation and attach to it immediately.
    log.findNext(re.escape("Acquired 1 IPU(s): running on device Id 2"))
    # Devices 0 & 1 are in use so we'll get device 2.
    log.findNext("Attached to device 2")
    log.findNext("Finished Poplar compilation")
