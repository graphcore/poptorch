#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import types
import poptorch
import torch.optim as optim


def test_weights_sharing_ipu_cpu():
    torch.manual_seed(42)
    model = torch.nn.Linear(10, 10)

    training_model = poptorch.trainingModel(model, loss=torch.nn.MSELoss())
    training_model.deviceToHostCounter = 0
    realMethod = training_model.copyWeightsToHost

    original_parameters = str([p for p in model.parameters()])

    def deviceToHostWrapper(model):
        model.deviceToHostCounter += 1
        realMethod()

    training_model.copyWeightsToHost = types.MethodType(
        deviceToHostWrapper, training_model)

    # Same model as above, they will share weights (in 'model') which once training is finished can be copied back.
    target = torch.randn(10)
    input = torch.randn(10)

    # Make sure the first run doesn't already pass the test.
    original, loss = training_model(input, target)
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    # Train on IPU.
    for i in range(0, 1000):
        out, loss = training_model(input, target)

    assert training_model.deviceToHostCounter == 0, "No implicit copy needed to train the model"

    # Run without copying the weights and check they've been automatically updated.
    nativeOut = model(input)
    assert torch.allclose(nativeOut, out)
    assert training_model.deviceToHostCounter == 1, "1 implicit copy after having trained the model"
    training_model.deviceToHostCounter = 0  # reset counter

    current_parameters = str([p for p in model.parameters()])
    assert original_parameters != current_parameters
    assert training_model.deviceToHostCounter == 0, "No implicit copy needed to access the parameters after inference"
    last_parameters = current_parameters

    nativeOut = model(input)
    assert torch.allclose(nativeOut, out)
    assert training_model.deviceToHostCounter == 0, "No implicit copy needed after inference"

    current_parameters = str([p for p in model.parameters()])
    assert last_parameters == current_parameters
    assert training_model.deviceToHostCounter == 0, "No implicit copy needed to access the parameters after inference"

    # Train on IPU.
    for i in range(0, 1000):
        out, loss = training_model(input, target)

    current_parameters = str([p for p in model.parameters()])
    assert training_model.deviceToHostCounter == 1, "1 implicit copy after having trained the model"
    assert original_parameters != current_parameters
    training_model.deviceToHostCounter = 0  # reset counter

    for i in range(0, 500):
        out, loss = training_model(input, target)
    assert training_model.deviceToHostCounter == 0, "No implicit copy needed to train the model"

    # Run without copying the weights and check they've been automatically updated.
    nativeOut = model(input)
    assert torch.allclose(nativeOut, out)
    assert training_model.deviceToHostCounter == 1, "1 implicit copy after having trained the model"
    training_model.deviceToHostCounter = 0  # reset counter

    nativeOut = model(input)
    assert torch.allclose(nativeOut, out)
    assert training_model.deviceToHostCounter == 0, "No implicit copy needed after inference"

    # Check we have trained the "model"
    assert torch.allclose(nativeOut, target, rtol=1e-02, atol=1e-02)


def test_weights_sharing_ipus():
    torch.manual_seed(42)
    model = torch.nn.Linear(10, 10)

    training_model = poptorch.trainingModel(model, loss=torch.nn.MSELoss())
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
    original, loss = training_model(input, target)
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    # Train on IPU.
    for i in range(0, 1000):
        out, loss = training_model(input, target)

    assert training_model.deviceToHostCounter == 0, "No implicit copy needed to train the model"

    # Run without copying the weights and check they've been automatically updated.
    out_inference = inference_model(input)
    assert torch.allclose(out_inference, out)
    assert training_model.deviceToHostCounter == 1, "1 implicit copy after having trained the model"
    training_model.deviceToHostCounter = 0  # reset counter

    out_inference = inference_model(input)
    assert torch.allclose(out_inference, out)
    assert training_model.deviceToHostCounter == 0, "No implicit copy needed after inference"

    # Train on IPU.
    for i in range(0, 1500):
        out, loss = training_model(input, target)

    assert training_model.deviceToHostCounter == 0, "No implicit copy needed to train the model"

    # Run without copying the weights and check they've been automatically updated.
    out_inference = inference_model(input)
    assert torch.allclose(out_inference, out)
    assert training_model.deviceToHostCounter == 1, "1 implicit copy after having trained the model"
    training_model.deviceToHostCounter = 0  # reset counter

    out_inference = inference_model(input)
    assert torch.allclose(out_inference, out)
    assert training_model.deviceToHostCounter == 0, "No implicit copy needed after inference"

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
    for i in range(0, 2500):
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
    for i in range(0, 2500):
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

    poptorch_model = poptorch.trainingModel(model, loss=torch.nn.MSELoss())
    target = torch.randn(10)
    input = torch.randn(10)

    # Make sure the first run doesn't already pass the test.
    original, loss = poptorch_model(input, target)
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    # Train on IPU.
    for i in range(0, 2500):
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
