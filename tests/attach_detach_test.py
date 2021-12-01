#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import math
import unittest.mock
import pytest
import torch
import helpers
import poptorch


@unittest.mock.patch.dict("os.environ", {"POPTORCH_WAIT_FOR_IPU": "0"})
@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed to test this feature")
@pytest.mark.parametrize("trace_model", [True, False])
def test_attach_detach(trace_model):
    torch.manual_seed(42)

    target = torch.randint(0, 10, [1])
    target = target.expand([10])
    input = torch.randn(10, 10)

    model = torch.nn.Linear(10, 10)

    opts = poptorch.Options()
    # Ensure that both models use the same IPU
    opts.useIpuId(1)

    training = helpers.trainingModelWithLoss(model,
                                             options=opts,
                                             loss=torch.nn.CrossEntropyLoss())

    opts = opts.clone()
    opts.Jit.traceModel(trace_model)
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


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed to test this feature")
@pytest.mark.parametrize("trace_model", [True, False])
def test_attach_detach_accuracy(trace_model):
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

    model_opts = poptorch.Options()
    model_opts.Jit.traceModel(trace_model)
    input_data = torch.Tensor([[1.], [-1.]])
    labels_data = torch.Tensor([0, 1]).long()
    model_with_loss = TrainingModelWithLoss()
    optimizer = poptorch.optim.SGD(model_with_loss.parameters(),
                                   lr=0.1,
                                   use_combined_accum=False)
    training_model = poptorch.trainingModel(model_with_loss,
                                            model_opts,
                                            optimizer=optimizer)
    inference_model = poptorch.inferenceModel(model_with_loss, model_opts)

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
