#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import copy
from io import StringIO
import json

import pytest
import torch
import torch.optim as optim
import helpers
import poptorch


# Convenience classes for testing
class LAMBNoBias(poptorch.optim.LAMB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, bias_correction=False, **kwargs)


class AdamWNoBias(poptorch.optim.AdamW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, bias_correction=False, **kwargs)


@pytest.mark.parametrize(
    "opt", {
        optim.SGD, optim.Adam, optim.AdamW, optim.RMSprop, poptorch.optim.SGD,
        poptorch.optim.Adam, poptorch.optim.AdamW, poptorch.optim.RMSprop,
        poptorch.optim.LAMB, AdamWNoBias, LAMBNoBias
    })
def test_optimizer(opt):
    torch.manual_seed(42)

    model = torch.nn.Linear(10, 10)

    # "Train" with learning rate of zero and check the loss remains the same.
    optimizer = opt(model.parameters(), lr=0.00)

    poptorch_model = helpers.trainingModelWithLoss(
        model, loss=torch.nn.CrossEntropyLoss(), optimizer=optimizer)

    input = torch.randn(1, 10)
    label = torch.randint(0, 10, [1])

    # Make sure the first run doesn't already pass the test.
    _, original_loss = poptorch_model(input, label)

    # Loss shouldn't change.
    for _ in range(0, 50):
        out, loss = poptorch_model(input, label)
        assert loss == original_loss

    # We shouldn't get the right result.
    assert not torch.argmax(out, dim=1) == label

    # Update the optimizer and check the loss now begins to decrease.
    optimizer.param_groups[0]['lr'] = 0.01
    poptorch_model.setOptimizer(optimizer)

    for _ in range(0, 1000):
        out, loss = poptorch_model(input, label)

    # Check we have trained the "model"
    assert loss < original_loss
    assert loss < 0.03
    assert torch.argmax(out, dim=1) == label


@pytest.mark.parametrize(
    "opt", {optim.SGD, optim.AdamW, poptorch.optim.SGD, poptorch.optim.AdamW})
def test_sgd_IR(opt):
    torch.manual_seed(42)
    model = torch.nn.Linear(10, 10)

    # "Train" with learning rate of zero and check the loss remains the same.
    optimizer = opt(model.parameters(), lr=0.01)

    poptorch_model = helpers.trainingModelWithLoss(
        model, loss=torch.nn.CrossEntropyLoss(), optimizer=optimizer)

    input = torch.randn(1, 10)
    label = torch.randint(0, 10, [1])

    poptorch_model(input, label)

    as_json = json.load(StringIO(poptorch_model._debugGetPopartIR()))  # pylint: disable=protected-access

    AdamVarUpdate = 0
    AdamUpdater = 0
    SGD0VarUpdate = 0
    for name in as_json:
        assert name == "maingraph"
        for op in as_json[name]:
            if op['type'] == "AdamUpdater":
                AdamUpdater += 1
            elif op['type'] == "AdamVarUpdate":
                AdamVarUpdate += 1
            elif op['type'] == "SGD0VarUpdate":
                SGD0VarUpdate += 1

    if opt in (optim.SGD, poptorch.optim.SGD):
        assert SGD0VarUpdate == 2
        assert AdamVarUpdate == 0 and AdamUpdater == 0
    else:
        assert SGD0VarUpdate == 0
        assert AdamVarUpdate == 2 and AdamUpdater == 2


@pytest.mark.parametrize("opt", (poptorch.optim.Adam, poptorch.optim.AdamW,
                                 AdamWNoBias, poptorch.optim.LAMB, LAMBNoBias))
@pytest.mark.parametrize("accType", (torch.float16, torch.float))
def test_IR_accum_type(opt, accType):
    torch.manual_seed(42)
    model = torch.nn.Linear(10, 10).half()

    # "Train" with learning rate of zero and check the loss remains the same.
    optimizer = opt(model.parameters(), lr=0.01, accum_type=accType)
    # These two should also be tested but they don't appear to work in popart yet.
    #firstOrderMomentumAccumType=torch.float16,
    #secondOrderMomentumAccumType=torch.float16 )
    poptorch_model = helpers.trainingModelWithLoss(
        model, loss=torch.nn.CrossEntropyLoss().half(), optimizer=optimizer)

    input = torch.randn(1, 10).half()
    label = torch.randint(0, 10, [1])

    poptorch_model(input, label)

    as_json = json.load(StringIO(poptorch_model._debugGetPopartIR()))  # pylint: disable=protected-access

    numCastsFound = sum([op["type"] == "Cast" for op in as_json["maingraph"]])

    if accType == torch.float16:
        assert numCastsFound == 2
    else:
        assert numCastsFound == 0


def test_velocity_scaling_copy():
    torch.manual_seed(42)

    model = torch.nn.Linear(10, 10)

    # "Train" with learning rate of zero and check the loss remains the same.
    optimizer = poptorch.optim.SGD(model.parameters(),
                                   lr=0.05,
                                   loss_scaling=0.05,
                                   velocity_scaling=128.1)

    poptorch_model = helpers.trainingModelWithLoss(
        model,
        loss=torch.nn.CrossEntropyLoss(reduction="sum"),
        optimizer=optimizer)

    input = torch.randn(1, 10)
    label = torch.randint(0, 10, [1])

    poptorch_model(input, label)

    # Check copy.copy preserves optimizer Poptorch attributes
    o = copy.copy(optimizer)
    poptorch_model.setOptimizer(o)
    poptorch_model(input, label)


def optimizer_groups_harness(opt, model, input, target):
    # Start the optimizer as zero for both groups.
    poptorch_model = poptorch.trainingModel(
        model,
        optimizer=opt([{
            'params': model.model[0].parameters(),
            "lr": 0.0
        }, {
            'params': model.model[1].parameters(),
            "lr": 0.0
        }],
                      lr=0.1))

    # Parameter is a soft copy by default oddly.
    weight1 = model.model[0].weight.clone()
    bias1 = model.model[0].bias.clone()
    weight2 = model.model[1].weight.clone()
    bias2 = model.model[1].bias.clone()

    _, original_loss = poptorch_model(input, target)
    for _ in range(0, 100):
        out, loss = poptorch_model(input, target)

    weight1_post, bias1_post = model.model[0].parameters()
    weight2_post, bias2_post = model.model[1].parameters()

    # Nothing should have changed.
    assert torch.equal(weight1, weight1_post)
    assert torch.equal(weight2, weight2_post)
    assert torch.equal(bias1, bias1_post)
    assert torch.equal(bias2, bias2_post)

    # Check we have not trained the model
    assert loss == original_loss

    # Now update the optimizer to train just one weight
    poptorch_model.setOptimizer(
        optimizer=opt([{
            'params': model.model[0].parameters(),
            "lr": 0.1
        }, {
            'params': model.model[1].parameters(),
            "lr": 0.0
        }],
                      lr=0.1))

    _, original_loss = poptorch_model(input, target)

    for _ in range(0, 100):
        out, loss = poptorch_model(input, target)

    weight1_post, bias1_post = model.model[0].parameters()
    weight2_post, bias2_post = model.model[1].parameters()

    assert loss != original_loss

    assert not torch.equal(weight1, weight1_post)
    assert torch.equal(weight2, weight2_post)
    assert not torch.equal(bias1, bias1_post)
    assert torch.equal(bias2, bias2_post)

    # Now update the optimizer to train just both weight
    poptorch_model.setOptimizer(
        optimizer=opt([{
            'params': model.model[0].parameters(),
            "lr": 0.1
        }, {
            'params': model.model[1].parameters(),
            "lr": 0.1
        }],
                      lr=0.1))

    _, original_loss = poptorch_model(input, target)

    # Actually try and train here.
    for _ in range(0, 2000):
        out, loss = poptorch_model(input, target)

    weight2_post, bias2_post = model.model[1].parameters()

    assert not torch.equal(weight2, weight2_post)
    assert not torch.equal(bias2, bias2_post)

    return out


@pytest.mark.parametrize("opt", {
    optim.SGD,
    poptorch.optim.SGD,
})
def test_optimizer_groups_sgd(opt):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                             torch.nn.Linear(10, 10),
                                             torch.nn.Sigmoid())
            self.loss = torch.nn.BCELoss()

        def forward(self, X, Y):
            fwd = self.model(X)
            return fwd, self.loss(fwd, Y)

    model = Model()

    target = torch.empty(10).uniform_()
    input = torch.randn(10)

    out = optimizer_groups_harness(opt, model, input, target)

    # Check we've trained the model.
    torch.testing.assert_allclose(target, out, rtol=1e-03, atol=1e-03)


@pytest.mark.parametrize(
    "opt", {
        optim.Adam, optim.AdamW, optim.RMSprop, poptorch.optim.Adam,
        poptorch.optim.AdamW, AdamWNoBias, poptorch.optim.RMSprop,
        poptorch.optim.LAMB, LAMBNoBias
    })
def test_optimizer_groups(opt):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                             torch.nn.Linear(10, 10))
            #torch.nn.Sigmoid())
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, X, Y):
            fwd = self.model(X)
            return fwd, self.loss(fwd, Y)

    model = Model()

    input = torch.randn(1, 10)
    target = torch.randint(0, 10, [1])

    out = optimizer_groups_harness(opt, model, input, target)

    # Check we've trained the model.
    assert torch.argmax(out) == target


def test_optimizer_groups_none_args():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                             torch.nn.Linear(10, 10))
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, X, Y, Z, B=None):  # pylint: disable=unused-argument
            fwd = self.model(X)
            return fwd, self.loss(fwd, Y)

    model = Model()

    input = torch.randn(1, 10)
    target = torch.randint(0, 10, [1])

    # Start the optimizer as zero for both groups.
    poptorch_model = poptorch.trainingModel(
        model,
        optimizer=optim.AdamW([{
            'params': model.model[0].parameters(),
            "lr": 0.0
        }, {
            'params': model.model[1].parameters(),
            "lr": 0.0
        }],
                              lr=0.1))

    poptorch_model.compile(input, target, target)

    # Parameter is a soft copy by default oddly.
    weight1 = model.model[0].weight.clone()
    bias1 = model.model[0].bias.clone()
    weight2 = model.model[1].weight.clone()
    bias2 = model.model[1].bias.clone()

    _, _ = poptorch_model(input, target, target)
    for _ in range(0, 100):
        _, _ = poptorch_model(input, target, target)

    weight1_post, bias1_post = model.model[0].parameters()
    weight2_post, bias2_post = model.model[1].parameters()

    # Nothing should have changed.
    assert torch.equal(weight1, weight1_post)
    assert torch.equal(weight2, weight2_post)
    assert torch.equal(bias1, bias1_post)
    assert torch.equal(bias2, bias2_post)


def test_optimizer_SGD_nesterov():
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                             torch.nn.Linear(10, 10))
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, X, Y):
            fwd = self.model(X)
            return fwd, self.loss(fwd, Y)

    model = Model()

    with pytest.raises(ValueError,
                       match="Nesterov momentum is currently not supported"):
        poptorch.trainingModel(model,
                               optimizer=optim.SGD(model.parameters(),
                                                   nesterov=True,
                                                   momentum=0.1,
                                                   lr=0.001))


@pytest.mark.parametrize(
    "opt", {
        poptorch.optim.SGD, poptorch.optim.Adam, poptorch.optim.AdamW,
        poptorch.optim.RMSprop, poptorch.optim.LAMB, AdamWNoBias, LAMBNoBias
    })
def test_optimizer_const(opt):
    torch.manual_seed(42)

    model = torch.nn.Linear(10, 10)
    # Initialise the optimiser with the default loss_scaling value
    optimizer = opt(model.parameters(), loss_scaling=1.0, lr=1.0)

    input = torch.randn(1, 10)
    label = torch.randint(0, 10, [1])
    poptorch_model = helpers.trainingModelWithLoss(
        model, loss=torch.nn.CrossEntropyLoss(), optimizer=optimizer)
    poptorch_model(input, label)

    optimizer.loss_scaling = 2.0
    poptorch_model.setOptimizer(optimizer)
    poptorch_model(input, label)


@pytest.mark.parametrize(
    "opt", {
        poptorch.optim.SGD, poptorch.optim.Adam, poptorch.optim.AdamW,
        poptorch.optim.RMSprop, poptorch.optim.LAMB, AdamWNoBias, LAMBNoBias
    })
def test_optimizer_mark_as_variable(opt):
    torch.manual_seed(42)

    model = torch.nn.Linear(10, 10)
    # Initialise the optimiser with the default loss_scaling value
    optimizer = opt(model.parameters(), lr=1.0)
    optimizer.variable_attrs.markAsVariable("loss_scaling")

    input = torch.randn(1, 10)
    label = torch.randint(0, 10, [1])
    poptorch_model = helpers.trainingModelWithLoss(
        model, loss=torch.nn.CrossEntropyLoss(), optimizer=optimizer)
    poptorch_model(input, label)

    optimizer.loss_scaling = 2.0
    poptorch_model.setOptimizer(optimizer)
    poptorch_model(input, label)


@pytest.mark.parametrize("opt", {poptorch.optim.LAMB, LAMBNoBias})
def test_lamb_max_weight_norm(opt):
    torch.manual_seed(42)

    model = torch.nn.Linear(10, 10)

    # With max_weight_norm=0.0, lr is multiplied by 0.0. The model shouldn't train.
    optimizer = opt(model.parameters(), lr=0.01, max_weight_norm=0.0)

    poptorch_model = helpers.trainingModelWithLoss(
        model, loss=torch.nn.CrossEntropyLoss(), optimizer=optimizer)

    input = torch.randn(1, 10)
    label = torch.randint(0, 10, [1])

    # Make sure the first run doesn't already pass the test.
    _, original_loss = poptorch_model(input, label)

    # Loss shouldn't change.
    for _ in range(0, 50):
        out, loss = poptorch_model(input, label)
        assert loss == original_loss

    # Update the optimizer with a non-zero max_weight_norm. It should now train.
    optimizer.max_weight_norm = 100.0
    poptorch_model.setOptimizer(optimizer)

    for _ in range(0, 1000):
        out, loss = poptorch_model(input, label)

    # Check we have trained the "model"
    assert loss < original_loss
    assert loss < 0.03
    assert torch.argmax(out, dim=1) == label

    # Run from scratch with max_weight_norm disabled.
    model = torch.nn.Linear(10, 10)
    optimizer = opt(model.parameters(), lr=0.01, max_weight_norm=None)
    poptorch_model = helpers.trainingModelWithLoss(
        model, loss=torch.nn.CrossEntropyLoss(), optimizer=optimizer)

    input = torch.randn(1, 10)
    label = torch.randint(0, 10, [1])

    # Train model again
    _, original_loss = poptorch_model(input, label)
    for _ in range(0, 1000):
        out, loss = poptorch_model(input, label)

    # Model should have trained like normal
    assert loss < original_loss
    assert loss < 0.03
    assert torch.argmax(out, dim=1) == label
