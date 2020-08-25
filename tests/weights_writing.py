#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import types

import poptorch
import pytest
import torch
import torch.optim as optim
import helpers


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
        assert len(loss.size()) == 1
        assert loss.size()[0] == 1

    # Run with trained weights.
    out = inference(input)

    # Check we are now equal with labels.
    assert torch.equal(torch.argmax(out.int(), dim=1), label)


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

    # Train on IPU.
    for _ in range(0, 1000):
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

    # Train on IPU.
    for _ in range(0, 1500):
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

    # Check we have trained the "model"
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
