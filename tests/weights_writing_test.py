#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import types
import copy
import numpy as np
import pytest
import torch
import torch.optim as optim
import helpers
import poptorch


@pytest.mark.parametrize("use_half", [True, False])
def test_training_and_inference(use_half):
    torch.manual_seed(42)

    # 10 Batches of 10.
    input = torch.randn(10, 10)

    # 10 batches of 1
    label = torch.randint(0, 10, [1])
    label = label.expand([10])
    model = torch.nn.Linear(10, 10)

    if use_half:
        model.half()
        input = input.half()

    # Run on IPU batch size 1 * 10 popart batches.
    opts = poptorch.Options().deviceIterations(10)
    poptorch_model = helpers.trainingModelWithLoss(
        model, options=opts, loss=torch.nn.CrossEntropyLoss())
    inference = poptorch.inferenceModel(model)

    # Run all 10 batches as batchsize 10.
    out = inference(input)

    # Sanity check we weren't already matching the label.
    assert not torch.equal(torch.argmax(out.int(), dim=1), label)

    for _ in range(0, 1000):
        _, loss = poptorch_model(input, label)

        # Each batch should NOT report its own loss. As by default training model should have a "Final" anchor.
        assert len(loss.size()) == 0

    # Run with trained weights.
    out = inference(input)

    # Check we are now equal with labels.
    assert torch.equal(torch.argmax(out.int(), dim=1), label)


@pytest.mark.parametrize("use_half", [True, False])
def test_training_inference_parameters(use_half):
    torch.manual_seed(42)

    # 10 Batches of 10.
    input = torch.randn(10, 10)

    # 10 batches of 1
    label = torch.randint(0, 10, [1])
    label = label.expand([10])
    model = torch.nn.Linear(10, 10)

    if use_half:
        model.half()
        input = input.half()

    # Run on IPU batch size 1 * 10 popart batches.
    opts = poptorch.Options().deviceIterations(10)
    poptorch_model = helpers.trainingModelWithLoss(
        model, options=opts, loss=torch.nn.CrossEntropyLoss())
    inference = poptorch.inferenceModel(model)

    # Run all 10 batches as batchsize 10.
    out = inference(input)

    # Sanity check we weren't already matching the label.
    assert not torch.equal(torch.argmax(out.int(), dim=1), label)

    for _ in range(0, 1000):
        _, loss = poptorch_model(input, label)

        # Each batch should NOT report its own loss. As by default training model should have a "Final" anchor.
        assert len(loss.size()) == 0

    # This will trigger copyWeightsToHost()
    for _ in model.named_parameters():
        pass

    # Run with trained weights.
    out = inference(input)

    # Check we are now equal with labels.
    assert torch.equal(torch.argmax(out.int(), dim=1), label)


@pytest.mark.parametrize("use_half", [True, False])
def test_access_parameters(use_half):
    torch.manual_seed(42)

    # 10 Batches of 10.
    input = torch.randn(10, 10)

    # 10 batches of 1
    label = torch.randint(0, 10, [1])
    label = label.expand([10])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    model = Model()

    if use_half:
        model.half()
        input = input.half()

    # Run on IPU batch size 1 * 10 popart batches.
    opts = poptorch.Options().deviceIterations(10)
    poptorch_model = helpers.trainingModelWithLoss(
        model, options=opts, loss=torch.nn.CrossEntropyLoss())

    original_weights = str(model.linear.weight)
    inference = poptorch.inferenceModel(model)

    # Run all 10 batches as batchsize 10.
    out = inference(input)

    assert original_weights == str(model.linear.weight)

    # Sanity check we weren't already matching the label.
    assert not torch.equal(torch.argmax(out.int(), dim=1), label)

    for _ in range(0, 1000):
        _, loss = poptorch_model(input, label)

        # Each batch should NOT report its own loss. As by default training model should have a "Final" anchor.
        assert len(loss.size()) == 0

    assert original_weights != str(poptorch_model.model.linear.weight)

    # Run with trained weights.
    out = inference(input)

    # Check we are now equal with labels.
    assert torch.equal(torch.argmax(out.int(), dim=1), label)


class DummyTrainingModel(torch.nn.Module):
    """
    Dummy training model
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 4, (3, 3))
        self.loss = torch.nn.NLLLoss()
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, target):
        x = self.conv(x)
        x = self.softmax(x)
        return self.loss(x, target)


def test_torch_save():
    torch.manual_seed(42)

    # create a dummy model
    model = DummyTrainingModel()

    # create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # store the weights before training
    pre_train_weights = copy.deepcopy(model.state_dict()['conv.weight'])

    # wrap it in a trainingModel
    training_model = poptorch.trainingModel(model, optimizer=optimizer)

    # run on dummy data for one iteration
    input = torch.randn(5, 16, 10, 10)
    target = torch.empty(5, 8, 8, dtype=torch.long).random_(0, 4)
    _ = training_model(input, target)

    # save the model
    torch.save(model, "/tmp/model.save")

    # reload the model
    reloaded_model = torch.load("/tmp/model.save")

    # make sure the reloaded weights are the same as the
    # model and trainingModel
    assert np.allclose(model.state_dict()['conv.weight'],
                       reloaded_model.state_dict()['conv.weight'])
    assert np.allclose(model.state_dict()['conv.weight'],
                       training_model.state_dict()['conv.weight'])

    # make sure we actually trained and we are not just checking
    # the original wrapped model weights
    assert not np.allclose(model.state_dict()['conv.weight'],
                           pre_train_weights)


def train_and_check_weight_sharing_ipu_cpu(model, training_model, input,
                                           target, original_parameters):
    # Make sure the first run doesn't already pass the test.
    original, _ = training_model(input, target)
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    # Train on IPU.
    for _ in range(0, 100):
        out, _ = training_model(input, target)

    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed to train the model"

    # Run without copying the weights and check they've been automatically updated.
    nativeOut = model(input)
    assert torch.allclose(nativeOut, out)
    assert training_model.deviceToHostCounter == 1, \
            "1 implicit copy after having trained the model"
    training_model.deviceToHostCounter = 0  # reset counter

    current_parameters = str(list(model.parameters()))
    assert original_parameters != current_parameters
    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed to access the parameters after inference"
    last_parameters = current_parameters

    nativeOut = model(input)
    assert torch.allclose(nativeOut, out)
    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed after inference"

    current_parameters = str(list(model.parameters()))
    assert last_parameters == current_parameters
    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed to access the parameters after inference"


def test_weights_sharing_ipu_cpu():
    torch.manual_seed(42)
    model = torch.nn.Linear(10, 10)

    training_model = helpers.trainingModelWithLoss(model,
                                                   loss=torch.nn.MSELoss())
    training_model.deviceToHostCounter = 0
    realMethod = training_model.copyWeightsToHost

    original_parameters = str(list(model.parameters()))

    def deviceToHostWrapper(model):
        model.deviceToHostCounter += 1
        realMethod()

    training_model.copyWeightsToHost = types.MethodType(
        deviceToHostWrapper, training_model)

    # Same model as above, they will share weights (in 'model') which once training is finished can be copied back.
    target = torch.randn(10)
    input = torch.randn(10)

    train_and_check_weight_sharing_ipu_cpu(model, training_model, input,
                                           target, original_parameters)

    # Train on IPU.
    for _ in range(0, 50):
        out, _ = training_model(input, target)

    current_parameters = str(list(model.parameters()))
    assert training_model.deviceToHostCounter == 1, \
            "1 implicit copy after having trained the model"
    assert original_parameters != current_parameters
    training_model.deviceToHostCounter = 0  # reset counter

    for _ in range(0, 50):
        out, _ = training_model(input, target)

    # Access a parameter directly:
    print(model.weight.data)

    assert training_model.deviceToHostCounter == 1, \
            "1 implicit copy after having trained the model"
    training_model.deviceToHostCounter = 0  # reset counter

    for _ in range(0, 50):
        out, _ = training_model(input, target)

    # Check state_dict works: torch.save(model.state_dict(), "/tmp/model.save")
    model.state_dict()

    assert training_model.deviceToHostCounter == 1, \
            "1 implicit copy after having trained the model"
    training_model.deviceToHostCounter = 0  # reset counter

    for _ in range(0, 50):
        out, _ = training_model(input, target)

    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed to train the model"

    # Run without copying the weights and check they've been automatically updated.
    nativeOut = model(input)
    assert torch.allclose(nativeOut, out)
    assert training_model.deviceToHostCounter == 1, \
            "1 implicit copy after having trained the model"
    training_model.deviceToHostCounter = 0  # reset counter

    nativeOut = model(input)
    assert torch.allclose(nativeOut, out)
    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed after inference"

    # Check we have trained the "model"
    assert torch.allclose(nativeOut, target, rtol=1e-02, atol=1e-02)


def train_N_times_and_check_copying(N, inference_model, training_model, input,
                                    target):
    # Train on IPU.
    for _ in range(0, N):
        out, _ = training_model(input, target)

    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed to train the model"

    # Run without copying the weights and check they've been automatically updated.
    out_inference = inference_model(input)
    assert torch.allclose(out_inference, out)
    assert training_model.deviceToHostCounter == 1, \
            "1 implicit copy after having trained the model"
    training_model.deviceToHostCounter = 0  # reset counter

    out_inference = inference_model(input)
    assert torch.allclose(out_inference, out)
    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed after inference"

    return out_inference


def test_weights_sharing_ipus():
    torch.manual_seed(42)
    model = torch.nn.Linear(10, 10)

    training_model = helpers.trainingModelWithLoss(model,
                                                   loss=torch.nn.MSELoss())
    training_model.deviceToHostCounter = 0
    realMethod = training_model.copyWeightsToHost

    def deviceToHostWrapper(model):
        model.deviceToHostCounter += 1
        realMethod()

    training_model.copyWeightsToHost = types.MethodType(
        deviceToHostWrapper, training_model)

    # Same model as above, they will share weights (in 'model') which once training is finished can be copied back.
    inference_model = poptorch.inferenceModel(model)
    target = torch.randn(10)
    input = torch.randn(10)

    out_inference = inference_model(input)
    assert not torch.allclose(out_inference, target, rtol=1e-02, atol=1e-02)

    # Make sure the first run doesn't already pass the test.
    original, _ = training_model(input, target)
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    train_N_times_and_check_copying(1000, inference_model, training_model,
                                    input, target)
    out_inference = train_N_times_and_check_copying(1500, inference_model,
                                                    training_model, input,
                                                    target)

    assert torch.allclose(out_inference, target, rtol=1e-02, atol=1e-02)


def test_implicit_first_time_copy():
    torch.manual_seed(42)

    # Train on host.
    model = torch.nn.Linear(10, 10)
    target = torch.randn(10)
    input = torch.randn(10)

    loss_function = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.eval()
    # Make sure the first run doesn't already pass the test.
    native = model(input)
    assert not torch.allclose(native, target, rtol=1e-02, atol=1e-02)

    model.train()
    for _ in range(0, 2500):
        optimizer.zero_grad()

        # Run model.
        outputs = model(input)

        # Back prop loss.
        loss = loss_function(target, outputs)
        loss.backward()
        optimizer.step()

    # Check the model is now trained
    model.eval()
    native = model(input)
    assert torch.allclose(native, target, rtol=1e-02, atol=1e-02)

    # Run on IPU.
    ipuModel = poptorch.inferenceModel(model)
    poptorchOut = ipuModel(input)

    # Check IPU returns same value as native without the weights explicitly being copied.
    assert torch.allclose(poptorchOut, native)
    assert torch.allclose(poptorchOut, target, rtol=1e-02, atol=1e-02)


def test_implicit_first_time_copy_negative():
    torch.manual_seed(42)

    # Train on host.
    model = torch.nn.Linear(10, 10)
    target = torch.randn(10)
    input = torch.randn(10)

    loss_function = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.eval()
    # Make sure the first run doesn't already pass the test.
    native = model(input)
    assert not torch.allclose(native, target, rtol=1e-02, atol=1e-02)

    # Run on IPU.
    poptorch_model = poptorch.inferenceModel(model)
    poptorchOut = poptorch_model(input)

    # Weights should be copied so check we are matching host but NOT the target.
    assert torch.allclose(poptorchOut, native)
    assert not torch.allclose(native, target, rtol=1e-02, atol=1e-02)

    model.train()
    for _ in range(0, 2500):
        optimizer.zero_grad()

        # Run model.
        outputs = model(input)

        # Back prop loss.
        loss = loss_function(target, outputs)
        loss.backward()
        optimizer.step()

    # Check the model is now trained
    model.eval()
    native = model(input)
    assert torch.allclose(native, target, rtol=1e-02, atol=1e-02)

    # Without recompilation or copying the weights check we are matching neither host nor the target.
    poptorchOut = poptorch_model(input)

    # Check IPU *does not* return the same value as native
    assert not torch.allclose(poptorchOut, native)
    assert not torch.allclose(poptorchOut, target, rtol=1e-02, atol=1e-02)


def test_weight_overwrite_trained_weight():
    torch.manual_seed(42)
    model = torch.nn.Linear(10, 10)

    poptorch_model = helpers.trainingModelWithLoss(model,
                                                   loss=torch.nn.MSELoss())
    target = torch.randn(10)
    input = torch.randn(10)

    # Make sure the first run doesn't already pass the test.
    original, loss = poptorch_model(input, target)
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    # Train on IPU.
    for _ in range(0, 2500):
        trained_out, trained_loss = poptorch_model(input, target)

    # Check we have trained the "model"
    assert torch.allclose(trained_out, target, rtol=1e-02, atol=1e-02)

    # Overwrite the trained weights with weights from host.
    poptorch_model.copyWeightsToDevice()

    # Don't train them.
    poptorch_model.setOptimizer(optim.SGD(model.parameters(), lr=0.0))

    out, loss = poptorch_model(input, target)
    host_out = model(input)

    # Check we are no longer trained.
    assert not torch.allclose(out, target, rtol=1e-02, atol=1e-02)
    assert not torch.allclose(loss, trained_loss)

    assert torch.allclose(host_out, out)


@pytest.mark.parametrize("use_half", [True, False])
def test_access_scalar_parameter(use_half):
    class ExampleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            x += 1

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

    model = ExampleModelWithCustomLoss()
    input = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([30.0, 40.0, 50.0])
    if use_half:
        model.half()
        input = input.half()
        target = target.half()
    poptorch_model = poptorch.trainingModel(model)
    original_bias = str(poptorch_model.model.model.bias)

    for _ in range(10):
        poptorch_model(input=input, target=target)

    updated_bias = str(poptorch_model.model.model.bias)
    assert original_bias != updated_bias

    poptorch_model.copyWeightsToHost()
    # Bias should already be up to date
    assert updated_bias == str(poptorch_model.model.model.bias)
