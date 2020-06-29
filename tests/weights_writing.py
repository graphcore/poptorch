#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import poptorch
import torch.optim as optim


def test_weight_write_to_host():
    torch.manual_seed(42)
    model = torch.nn.Linear(10, 10)

    poptorch_model = poptorch.trainingModel(model,
                                            device_iterations=1,
                                            loss=torch.nn.MSELoss())
    target = torch.randn(10)
    input = torch.randn(10)

    # Make sure the first run doesn't already pass the test.
    original, loss = poptorch_model(input, target)
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    # Train on IPU.
    for i in range(0, 2500):
        out, loss = poptorch_model(input, target)

    # Run without copying the weights.
    nativeOut = model(input)
    assert not torch.allclose(nativeOut, out)

    # Copy weights
    poptorch_model.copyWeightsToHost()

    # Run again with the now trained weights.
    nativeOut = model(input)
    assert torch.allclose(nativeOut, out)

    # Check we have trained the "model"
    assert torch.allclose(nativeOut, target, rtol=1e-02, atol=1e-02)


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

    poptorch_model = poptorch.trainingModel(model,
                                            device_iterations=1,
                                            loss=torch.nn.MSELoss())
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
