#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
import types
import copy
import tempfile
import unittest.mock

import numpy as np
import pytest
import torch
import torch.optim as optim
import helpers
import poptorch


class ModelWithLoss(torch.nn.Module):
    def __init__(self, loss, use_dropout=False):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.loss = loss
        if use_dropout:
            self.dropout = torch.nn.Dropout()
        else:
            self.dropout = lambda x: x

    def forward(self, data, target=None):
        out = self.dropout(self.linear(data))

        if target is None:
            return out

        loss = self.loss(out, target)
        return out, loss


@pytest.mark.parametrize("use_half", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_training_and_inference(use_half, trace_model):
    torch.manual_seed(42)

    # 10 Batches of 10.
    input = torch.randn(10, 10)

    # 10 batches of 1
    label = torch.randint(0, 10, [1])
    label = label.expand([10])
    model = ModelWithLoss(torch.nn.CrossEntropyLoss())

    if use_half:
        model.half()
        input = input.half()

    # Run on IPU batch size 1 * 10 popart batches.
    opts = poptorch.Options().deviceIterations(10)
    opts.Jit.traceModel(trace_model)

    training = poptorch.trainingModel(model, options=opts)
    inference = poptorch.inferenceModel(model, options=opts)

    # Run all 10 batches as batchsize 10.
    out = inference(input)

    # Sanity check we weren't already matching the label.
    assert not torch.equal(torch.argmax(out.int(), dim=1), label)

    for _ in range(0, 1000):
        _, loss = training(input, label)

        # Each batch should NOT report its own loss. As by default training
        # model should have a "Final" output mode.
        assert len(loss.size()) == 0

    # Run with trained weights.
    out = inference(input)

    # Check we are now equal with labels.
    helpers.assert_allequal(actual=torch.argmax(out.int(), dim=1),
                            expected=label)


@pytest.mark.parametrize("use_half", [True, False])
@pytest.mark.mlirSupportRequired
def test_training_inference_parameters(use_half):
    torch.manual_seed(42)

    # 10 Batches of 10.
    input = torch.randn(10, 10)

    # 10 batches of 1
    label = torch.randint(0, 10, [1])
    label = label.expand([10])

    model = ModelWithLoss(torch.nn.CrossEntropyLoss())

    if use_half:
        model.half()
        input = input.half()

    # Run on IPU batch size 1 * 10 popart batches.
    opts = poptorch.Options().deviceIterations(10)
    opts.Jit.traceModel(False)
    inference = poptorch.inferenceModel(model, opts)
    training = poptorch.trainingModel(model, options=opts)
    inference = poptorch.inferenceModel(model)

    # Run all 10 batches as batchsize 10.
    out = inference(input)

    # Sanity check we weren't already matching the label.
    assert not torch.equal(torch.argmax(out.int(), dim=1), label)

    for _ in range(0, 1000):
        _, loss = training(input, label)

        # Each batch should NOT report its own loss. As by default training model should have a "Final" output mode.
        assert len(loss.size()) == 0

    # This will trigger copyWeightsToHost()
    for _ in model.named_parameters():
        pass

    # Run with trained weights.
    out = inference(input)

    # Check we are now equal with labels.
    helpers.assert_allequal(actual=torch.argmax(out.int(), dim=1),
                            expected=label)


@pytest.mark.parametrize("use_half", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_access_parameters(use_half, trace_model):
    torch.manual_seed(42)

    # 10 Batches of 10.
    input = torch.randn(10, 10)

    # 10 batches of 1
    label = torch.randint(0, 10, [1])
    label = label.expand([10])

    model = ModelWithLoss(torch.nn.CrossEntropyLoss())

    if use_half:
        model.half()
        input = input.half()

    # Run on IPU batch size 1 * 10 popart batches.
    opts = poptorch.Options().deviceIterations(10)
    opts.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=opts)

    original_weights = str(model.linear.weight)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference = poptorch.inferenceModel(model, options)

    # Run all 10 batches as batchsize 10.
    out = inference(input)

    assert original_weights == str(model.linear.weight)

    # Sanity check we weren't already matching the label.
    assert not torch.equal(torch.argmax(out.int(), dim=1), label)

    for _ in range(0, 1000):
        _, loss = poptorch_model(input, label)

        # Each batch should NOT report its own loss. As by default training model should have a "Final" output mode.
        assert len(loss.size()) == 0

    assert original_weights != str(poptorch_model.model.linear.weight)

    # Run with trained weights.
    out = inference(input)

    # Check we are now equal with labels.
    helpers.assert_allequal(actual=torch.argmax(out.int(), dim=1),
                            expected=label)


class DummyTrainingModel(torch.nn.Module):
    """
    Dummy training model
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 4, (3, 3))
        self.loss = torch.nn.NLLLoss()
        self.batch_norm = torch.nn.BatchNorm2d(4)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, target):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.softmax(x)
        return self.loss(x, target)


@pytest.mark.parametrize("trace_model", [True, False])
def test_torch_save(trace_model):
    if not trace_model:
        pytest.skip(
            "TODO(T51159): NotImplementedError: Cannot access storage of "
            "IpuTensorImpl")
    torch.manual_seed(42)

    # create a dummy model
    model = DummyTrainingModel()

    # create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # store the weights before training
    pre_train_weights = copy.deepcopy(model.state_dict()['conv.weight'])

    # wrap it in a trainingModel
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    training_model = poptorch.trainingModel(model,
                                            options=options,
                                            optimizer=optimizer)

    # run on dummy data for one iteration
    input = torch.randn(5, 16, 10, 10)
    target = torch.empty(5, 8, 8, dtype=torch.long).random_(0, 4)
    _ = training_model(input, target)

    with tempfile.TemporaryDirectory(dir=".") as d:
        model_file = os.path.join(d, "model.save")
        # save the model
        torch.save(model, model_file)

        # reload the model
        reloaded_model = torch.load(model_file)

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


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_seed_precompilation(capfd, trace_model):
    # create a dummy model
    model = ModelWithLoss(torch.nn.CrossEntropyLoss(), use_dropout=True)

    # create optimizer
    optimizer = poptorch.optim.SGD(model.parameters(), lr=0.01)

    opts = poptorch.Options().randomSeed(42)
    opts.useOfflineIpuTarget(poptorch.ipuHardwareVersion())
    opts.Jit.traceModel(trace_model)
    training_model = poptorch.trainingModel(model,
                                            options=opts,
                                            optimizer=optimizer)
    # 10 Batches of 10.
    input = torch.randn(10, 10)

    # 10 batches of 1
    label = torch.randint(0, 10, [1])
    label = label.expand([10])
    training_model.compile(input, label)

    # Clear the outputs (We only want to parse what's triggered by save()
    helpers.LogChecker(capfd)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "checkpoint.pt")
        training_model.save(path)

        # Creating a checkpoint should trigger copies of the weights, optimizer state
        # random seed and rng state but as we're using an offline target nothing
        # should happen.
        log = helpers.LogChecker(capfd)
        log.assert_no_matches("Reading random seed")
        log.assert_no_matches("Reading RNG state")
        log.assert_no_matches("Implicit copyWeightsToHost()")
        log.assert_no_matches(
            "Writing optimiser state tensors from IPU to host.")

        poptorch.load(path)

        log = helpers.LogChecker(capfd)
        log.assert_matches("Writing weights from host to IPU memory")
        log.assert_matches("Setting random seed to")
        # We haven't run on HW so we don't have a state yet
        log.assert_no_matches("Setting RNG state")
        log.assert_no_matches(
            "Writing optimiser state tensors from host to IPU memory")


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.ipuHardwareRequired
@pytest.mark.parametrize("trace_model", [True, False])
def test_save_everything(capfd, trace_model):
    # create a dummy model
    model = ModelWithLoss(torch.nn.CrossEntropyLoss(), use_dropout=True)

    # create optimizer
    optimizer = poptorch.optim.SGD(model.parameters(), lr=0.01)

    opts = poptorch.Options().randomSeed(42)
    opts.Jit.traceModel(trace_model)
    training_model = poptorch.trainingModel(model,
                                            options=opts,
                                            optimizer=optimizer)
    # 10 Batches of 10.
    input = torch.randn(10, 10)

    # 10 batches of 1
    label = torch.randint(0, 10, [1])
    label = label.expand([10])
    first_out, first_loss = training_model(input, label)

    # Clear the outputs (We only want to parse what's triggered by save()
    helpers.LogChecker(capfd)

    origin_out = []
    loaded_out = []
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "checkpoint.pt")
        training_model.save(path)

        # Creating a checkpoint should trigger copies of the weights, optimizer state
        # random seed and rng state.
        log = helpers.LogChecker(capfd)
        log.assert_matches("Reading random seed")
        log.assert_matches("Reading RNG state")
        log.assert_matches("Implicit copyWeightsToHost()")
        log.assert_matches("Writing optimiser state tensors from IPU to host.")

        origin_out.append(training_model(input, label))

        loaded = poptorch.load(path)

        log = helpers.LogChecker(capfd)
        log.assert_matches("Writing weights from host to IPU memory")
        log.assert_matches("Setting random seed to")
        log.assert_matches("Setting RNG state")
        log.assert_matches(
            "Writing optimiser state tensors from host to IPU memory")

        loaded_out.append(loaded(input, label))
        origin_out.append(training_model(input, label))
        # Everything is loaded: there shouldn't be any transfer
        log = helpers.LogChecker(capfd)
        log.assert_no_matches("Writing weights from host to IPU memory")
        log.assert_no_matches("Implicit copyWeightsToHost()")
        log.assert_no_matches("random seed")
        log.assert_no_matches("RNG state")
        log.assert_no_matches("Writing optimiser state tensors from")

        loaded.detachFromDevice()
        log = helpers.LogChecker(capfd)
        log.assert_matches("Writing weights from IPU to host")
        log.assert_matches("Writing optimiser state tensors from IPU to host")
        log.assert_matches("Reading random seed")
        log.assert_matches("Reading RNG state")
        log.assert_matches("Detached from device")

        loaded_out.append(loaded(input, label))
        log = helpers.LogChecker(capfd)
        log.assert_matches("Writing weights from host to IPU memory")
        log.assert_matches(
            "Writing optimiser state tensors from host to IPU memory")
        log.assert_matches("Setting random seed to")
        log.assert_matches("Setting RNG state")

    for (out, loss), (load_out, load_loss) in zip(origin_out, loaded_out):
        helpers.assert_allclose(expected=out, actual=load_out)
        assert loss == load_loss
        assert not torch.allclose(out, first_out, rtol=1e-02, atol=1e-02)
        assert loss != first_loss


def train_and_check_weight_sharing_ipu_cpu(model, training_model, input,
                                           target, original_parameters):
    # Make sure the first run doesn't already pass the test.
    original, _ = training_model(input, target)
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    # Train on IPU.
    for _ in range(0, 1000):
        out, _ = training_model(input, target)

    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed to train the model"

    # Run without copying the weights and check they've been automatically updated.
    native_out = model(input)
    helpers.assert_allclose(expected=native_out, actual=out)
    assert training_model.deviceToHostCounter == 1, \
            "1 implicit copy after having trained the model"
    training_model.deviceToHostCounter = 0  # reset counter

    current_parameters = str(list(model.parameters()))
    assert original_parameters != current_parameters
    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed to access the parameters after inference"
    last_parameters = current_parameters

    native_out = model(input)
    helpers.assert_allclose(expected=native_out, actual=out)
    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed after inference"

    current_parameters = str(list(model.parameters()))
    assert last_parameters == current_parameters
    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed to access the parameters after inference"


@pytest.mark.parametrize("trace_model", [True, False])
def test_weights_sharing_ipu_cpu(trace_model):
    torch.manual_seed(42)

    model = ModelWithLoss(torch.nn.MSELoss())

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    training_model = poptorch.trainingModel(model, options=options)

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
    print(model.linear.weight.data)

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
    native_out = model(input)
    helpers.assert_allclose(expected=native_out, actual=out)
    assert training_model.deviceToHostCounter == 1, \
            "1 implicit copy after having trained the model"
    training_model.deviceToHostCounter = 0  # reset counter

    native_out = model(input)
    helpers.assert_allclose(expected=native_out, actual=out)
    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed after inference"

    # Check we have trained the "model"
    helpers.assert_allclose(expected=native_out,
                            actual=target,
                            rtol=1e-02,
                            atol=1e-02)


def train_N_times_and_check_copying(N, inference_model, training_model, input,
                                    target):
    # Train on IPU.
    for _ in range(0, N):
        out, _ = training_model(input, target)

    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed to train the model"

    # Run without copying the weights and check they've been automatically updated.
    out_inference = inference_model(input)
    helpers.assert_allclose(expected=out, actual=out_inference)
    assert training_model.deviceToHostCounter == 1, \
            "1 implicit copy after having trained the model"
    training_model.deviceToHostCounter = 0  # reset counter

    out_inference = inference_model(input)
    helpers.assert_allclose(expected=out, actual=out_inference)
    assert training_model.deviceToHostCounter == 0, \
            "No implicit copy needed after inference"

    return out_inference


@pytest.mark.parametrize("trace_model", [True, False])
def test_weights_sharing_ipus(trace_model):
    torch.manual_seed(42)

    model = ModelWithLoss(torch.nn.MSELoss())

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    training_model = poptorch.trainingModel(model, options)

    training_model.deviceToHostCounter = 0
    realMethod = training_model.copyWeightsToHost

    def deviceToHostWrapper(model):
        model.deviceToHostCounter += 1
        realMethod()

    training_model.copyWeightsToHost = types.MethodType(
        deviceToHostWrapper, training_model)

    # Same model as above, they will share weights (in 'model') which once training is finished can be copied back.
    inference_model = poptorch.inferenceModel(model, options)
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

    helpers.assert_allclose(actual=out_inference,
                            expected=target,
                            rtol=1e-02,
                            atol=1e-02)


@pytest.mark.parametrize("trace_model", [True, False])
def test_implicit_first_time_copy(trace_model):
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
    helpers.assert_allclose(actual=native,
                            expected=target,
                            rtol=1e-02,
                            atol=1e-02)

    # Run on IPU.
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    ipuModel = poptorch.inferenceModel(model, options)
    poptorch_out = ipuModel(input)

    # Check IPU returns same value as native without the weights explicitly being copied.
    helpers.assert_allclose(expected=native, actual=poptorch_out)
    helpers.assert_allclose(actual=poptorch_out,
                            expected=target,
                            rtol=1e-02,
                            atol=1e-02)


@pytest.mark.parametrize("trace_model", [True, False])
def test_implicit_first_time_copy_negative(trace_model):
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
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    poptorch_out = poptorch_model(input)

    # Weights should be copied so check we are matching host but NOT the target.
    helpers.assert_allclose(expected=native, actual=poptorch_out)
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
    helpers.assert_allclose(actual=native,
                            expected=target,
                            rtol=1e-02,
                            atol=1e-02)

    # Without recompilation or copying the weights check we are matching neither host nor the target.
    poptorch_out = poptorch_model(input)

    # Check IPU *does not* return the same value as native
    assert not torch.allclose(poptorch_out, native)
    assert not torch.allclose(poptorch_out, target, rtol=1e-02, atol=1e-02)


@pytest.mark.parametrize("trace_model", [True, False])
def test_weight_overwrite_trained_weight(trace_model):
    torch.manual_seed(42)

    model = ModelWithLoss(torch.nn.MSELoss())

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)

    target = torch.randn(10)
    input = torch.randn(10)

    # Make sure the first run doesn't already pass the test.
    original, loss = poptorch_model(input, target)
    assert not torch.allclose(original, target, rtol=1e-02, atol=1e-02)

    # Train on IPU.
    for _ in range(0, 2500):
        trained_out, trained_loss = poptorch_model(input, target)

    # Check we have trained the "model"
    helpers.assert_allclose(actual=trained_out,
                            expected=target,
                            rtol=1e-02,
                            atol=1e-02)

    # Overwrite the trained weights with weights from host.
    poptorch_model.copyWeightsToDevice()

    # Don't train them.
    poptorch_model.setOptimizer(optim.SGD(model.parameters(), lr=0.0))

    out, loss = poptorch_model(input, target)
    host_out = model(input)

    # Check we are no longer trained.
    assert not torch.allclose(out, target, rtol=1e-02, atol=1e-02)
    assert not torch.allclose(loss, trained_loss)

    helpers.assert_allclose(expected=host_out, actual=out)


@pytest.mark.parametrize("use_half", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_access_scalar_parameter(use_half, trace_model):
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

    model = ExampleModelWithCustomLoss()
    input = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([30.0, 40.0, 50.0])
    if use_half:
        model.half()
        input = input.half()
        target = target.half()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options=options)
    original_bias = str(poptorch_model.model.model.bias)

    for _ in range(10):
        poptorch_model(input=input, target=target)

    updated_bias = str(poptorch_model.model.model.bias)
    assert original_bias != updated_bias

    poptorch_model.copyWeightsToHost()
    # Bias should already be up to date
    assert updated_bias == str(poptorch_model.model.model.bias)


@pytest.mark.parametrize("reverse_equal_call", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_copy_on_torch_equal(reverse_equal_call, trace_model):
    torch.manual_seed(42)

    model = ModelWithLoss(torch.nn.MSELoss())

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model,
                                            options=options,
                                            optimizer=torch.optim.SGD(
                                                model.parameters(), lr=0.01))

    target = torch.ones(10)
    input = torch.randn(10)

    weight_at_start = model.linear.weight.clone().data

    for _ in range(100):
        poptorch_model(input, target)

    if reverse_equal_call:
        assert not torch.equal(model.linear.weight, weight_at_start)
    else:
        assert not torch.equal(weight_at_start, model.linear.weight)


@pytest.mark.parametrize("trace_model", [True, False])
def test_copy_after_compile(trace_model):
    torch.manual_seed(42)

    model = ModelWithLoss(torch.nn.MSELoss())

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model,
                                            options=options,
                                            optimizer=torch.optim.SGD(
                                                model.parameters(), lr=0.01))

    target = torch.ones(10)
    input = torch.randn(10)

    poptorch_model.compile(input, target)

    # If we haven't copied the weights, Popart will fire an exception
    # when trying to execute the model.
    poptorch_model(input, target)


@pytest.mark.parametrize("trace_model", [True, False])
def test_torch_save_unwrapped(trace_model):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 2, 1, padding=0)
            self.register_buffer("test_buffer",
                                 torch.zeros([2], dtype=torch.float32))
            self.register_parameter("test_param",
                                    torch.nn.Parameter(torch.empty(10)))
            self.loss = torch.nn.L1Loss()

        def forward(self, inp):
            out = self.conv(inp)
            loss = self.loss(out)
            return out, loss

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    # Only training models instrument the model so we can't use poptporch.inferenceModel
    poptorch.trainingModel(model, options)

    # An inference model sharing its user model with a training model will be instrumented though.
    poptorch.inferenceModel(model, options)

    with tempfile.TemporaryDirectory() as tmp:
        torch_file = os.path.join(tmp, "torch_saved.pt")
        torch.save(model.state_dict(), torch_file)

        # Ensure the state dictionaries returned by the training and inference models don't contain any PopTorch wrapper.
        with unittest.mock.patch.object(
                poptorch._impl,  # pylint: disable=protected-access
                "_pickleRestoreWrapperIfPossible",
                wraps=poptorch._impl._pickleRestoreWrapperIfPossible  # pylint: disable=protected-access
        ) as restore_fn:
            torch.load(torch_file)
            restore_fn.assert_not_called()
