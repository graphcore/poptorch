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


class OptimizerTestModel:
    def __init__(self, num_groups=1):
        layers = [torch.nn.Linear(10, 10) for _ in range(num_groups)]
        if num_groups == 1:
            self.model = layers[0]
        else:
            self.model = torch.nn.Sequential(*layers)
        self.input = torch.randn(1, 10)
        self.label = torch.randint(0, 10, [1])
        self.poptorch_model = None

    def parameters(self):
        return self.model.parameters()

    def setOptimizer(self, optimizer):
        self.poptorch_model.setOptimizer(optimizer)

    def run(self, optimizer=None):
        if not self.poptorch_model:
            assert optimizer, ("An optimizer must be provided to compile "
                               "the model")
            self.poptorch_model = helpers.trainingModelWithLoss(
                self.model,
                loss=torch.nn.CrossEntropyLoss(),
                optimizer=optimizer)
        elif optimizer:
            self.setOptimizer(optimizer)
        return self.poptorch_model(self.input, self.label)


@pytest.mark.parametrize(
    "opt", {
        optim.SGD, optim.Adam, optim.AdamW, optim.RMSprop, poptorch.optim.SGD,
        poptorch.optim.Adam, poptorch.optim.AdamW, poptorch.optim.RMSprop,
        poptorch.optim.LAMB, AdamWNoBias, LAMBNoBias
    })
def test_optimizer(opt):
    torch.manual_seed(42)

    model = OptimizerTestModel()

    # "Train" with learning rate of zero and check the loss remains the same.
    optimizer = opt(model.parameters(), lr=0.00)

    # Make sure the first run doesn't already pass the test.
    _, original_loss = model.run(optimizer)

    # Loss shouldn't change.
    for _ in range(0, 50):
        out, loss = model.run()
        assert loss == original_loss

    # We shouldn't get the right result.
    assert not torch.argmax(out, dim=1) == model.label

    # Update the optimizer and check the loss now begins to decrease.
    optimizer.param_groups[0]['lr'] = 0.01
    model.setOptimizer(optimizer)

    for _ in range(0, 1000):
        out, loss = model.run()

    # Check we have trained the "model"
    assert loss < original_loss
    assert loss < 0.03
    assert torch.argmax(out, dim=1) == model.label


@pytest.mark.parametrize(
    "opt", {optim.SGD, optim.AdamW, poptorch.optim.SGD, poptorch.optim.AdamW})
def test_sgd_IR(opt):
    torch.manual_seed(42)
    model = OptimizerTestModel()

    # "Train" with learning rate of zero and check the loss remains the same.
    optimizer = opt(model.parameters(), lr=0.01)

    model.run(optimizer)

    as_json = json.load(StringIO(model.poptorch_model._debugGetPopartIR()))  # pylint: disable=protected-access

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
    model = OptimizerTestModel()

    # "Train" with learning rate of zero and check the loss remains the same.
    optimizer = opt(model.parameters(), lr=0.01, accum_type=accType)
    # These two should also be tested but they don't appear to work in popart yet.
    # first_order_momentum_accum_type=torch.float16,
    # second_order_momentum_accum_type=torch.float16,
    #TODO ^
    model.run(optimizer)

    as_json = json.load(StringIO(model.poptorch_model._debugGetPopartIR()))  # pylint: disable=protected-access

    numCastsFound = sum([op["type"] == "Cast" for op in as_json["maingraph"]])

    if accType == torch.float16:
        assert numCastsFound == 2
    else:
        assert numCastsFound == 0


def test_velocity_scaling_copy():
    torch.manual_seed(42)

    model = OptimizerTestModel()

    # "Train" with learning rate of zero and check the loss remains the same.
    optimizer = poptorch.optim.SGD(model.parameters(),
                                   lr=0.05,
                                   loss_scaling=0.05,
                                   velocity_scaling=128.1)

    model.run(optimizer)

    # Check copy.copy preserves optimizer Poptorch attributes
    o = copy.copy(optimizer)
    model.setOptimizer(o)
    model.run()


@pytest.mark.parametrize(
    "opt", {
        optim.SGD, poptorch.optim.SGD, optim.Adam, optim.AdamW, optim.RMSprop,
        poptorch.optim.Adam, poptorch.optim.AdamW, AdamWNoBias,
        poptorch.optim.RMSprop, poptorch.optim.LAMB, LAMBNoBias
    })
def test_optimizer_groups(opt):
    torch.manual_seed(42)

    model = OptimizerTestModel(num_groups=2)

    # Parameter is a soft copy by default oddly.
    weight1 = model.model[0].weight.clone()
    bias1 = model.model[0].bias.clone()
    weight2 = model.model[1].weight.clone()
    bias2 = model.model[1].bias.clone()

    # Start the optimizer as zero for both groups.
    _, original_loss = model.run(
        opt([{
            'params': model.model[0].parameters(),
            "lr": 0.0
        }, {
            'params': model.model[1].parameters(),
            "lr": 0.0
        }],
            lr=0.1))

    for _ in range(0, 10):
        out, loss = model.run()

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
    _, original_loss = model.run(
        opt([{
            'params': model.model[0].parameters(),
            "lr": 0.1
        }, {
            'params': model.model[1].parameters(),
            "lr": 0.0
        }],
            lr=0.1))

    for _ in range(0, 10):
        out, loss = model.run()

    weight1_post, bias1_post = model.model[0].parameters()
    weight2_post, bias2_post = model.model[1].parameters()

    assert loss != original_loss

    assert not torch.equal(weight1, weight1_post)
    assert torch.equal(weight2, weight2_post)
    assert not torch.equal(bias1, bias1_post)
    assert torch.equal(bias2, bias2_post)

    # Now update the optimizer to train just both weight
    _, original_loss = model.run(
        opt([{
            'params': model.model[0].parameters(),
            "lr": 0.1
        }, {
            'params': model.model[1].parameters(),
            "lr": 0.1
        }],
            lr=0.1))

    # Actually try and train here.
    for _ in range(0, 2000):
        out, loss = model.run()

    weight2_post, bias2_post = model.model[1].parameters()

    assert not torch.equal(weight2, weight2_post)
    assert not torch.equal(bias2, bias2_post)

    # Check we've trained the model.
    assert torch.argmax(out) == model.label


def test_optimizer_groups_none_args():
    torch.manual_seed(42)

    model = OptimizerTestModel(num_groups=2)

    # Parameter is a soft copy by default oddly.
    weight1 = model.model[0].weight.clone()
    bias1 = model.model[0].bias.clone()
    weight2 = model.model[1].weight.clone()
    bias2 = model.model[1].bias.clone()

    # Start the optimizer as zero for both groups.
    model.run(
        optim.AdamW([{
            'params': model.model[0].parameters(),
            "lr": 0.0
        }, {
            'params': model.model[1].parameters(),
            "lr": 0.0
        }],
                    lr=0.1))

    for _ in range(0, 10):
        model.run()

    weight1_post, bias1_post = model.model[0].parameters()
    weight2_post, bias2_post = model.model[1].parameters()

    # Nothing should have changed.
    assert torch.equal(weight1, weight1_post)
    assert torch.equal(weight2, weight2_post)
    assert torch.equal(bias1, bias1_post)
    assert torch.equal(bias2, bias2_post)


def test_optimizer_SGD_nesterov():
    torch.manual_seed(42)
    model = OptimizerTestModel()

    with pytest.raises(ValueError,
                       match="Nesterov momentum is currently not supported"):
        model.run(
            optim.SGD(model.parameters(),
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

    model = OptimizerTestModel()

    # Initialise the optimiser with the default loss_scaling value
    optimizer = opt(model.parameters(), loss_scaling=1.0, lr=1.0)

    model.run(optimizer)

    optimizer.loss_scaling = 2.0
    model.run(optimizer)


@pytest.mark.parametrize(
    "opt", {
        poptorch.optim.SGD, poptorch.optim.Adam, poptorch.optim.AdamW,
        poptorch.optim.RMSprop, poptorch.optim.LAMB, AdamWNoBias, LAMBNoBias
    })
def test_optimizer_mark_as_variable(opt):
    torch.manual_seed(42)

    model = OptimizerTestModel()
    # Initialise the optimiser with the default loss_scaling value
    optimizer = opt(model.parameters(), lr=1.0)
    optimizer.variable_attrs.markAsVariable("loss_scaling")
    model.run(optimizer)

    optimizer.loss_scaling = 2.0
    model.run(optimizer)


@pytest.mark.parametrize("opt", {poptorch.optim.LAMB, LAMBNoBias})
def test_lamb_max_weight_norm(opt):
    torch.manual_seed(42)
    model = OptimizerTestModel()

    # With max_weight_norm=0.0, lr is multiplied by 0.0. The model shouldn't train.
    optimizer = opt(model.parameters(), lr=0.01, max_weight_norm=0.0)

    # Make sure the first run doesn't already pass the test.
    _, original_loss = model.run(optimizer)

    # Loss shouldn't change.
    for _ in range(0, 50):
        out, loss = model.run()
        assert loss == original_loss

    # Update the optimizer with a non-zero max_weight_norm. It should now train.
    optimizer.max_weight_norm = 100.0
    model.run(optimizer)

    for _ in range(0, 1000):
        out, loss = model.run()

    # Check we have trained the "model"
    assert loss < original_loss
    assert loss < 0.03
    assert torch.argmax(out, dim=1) == model.label

    # Run from scratch with max_weight_norm disabled.
    model = OptimizerTestModel()
    optimizer = opt(model.parameters(), lr=0.01, max_weight_norm=None)

    # Train model again
    _, original_loss = model.run(optimizer)
    for _ in range(0, 1000):
        out, loss = model.run()

    # Model should have trained like normal
    assert loss < original_loss
    assert loss < 0.03
    assert torch.argmax(out, dim=1) == model.label


@helpers.printCapfdOnExit
def test_variable_groups(capfd):
    poptorch.setLogLevel(1)  # Force debug logging
    model = OptimizerTestModel(num_groups=2)

    # Make sure all groups have the default values, and the values are not (const)
    params = [{
        "params": model.model[0].parameters()
    }, {
        "params": model.model[1].parameters()
    }]
    o = poptorch.optim.SGD(params,
                           lr=0.01,
                           loss_scaling=2.0,
                           velocity_scaling=2.0)
    model.run(o)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_contains("graph optimizer with SGD",
                            "defaultLearningRate=0.01,",
                            "defaultVelocityScaling=2,", "lossScaling=2")
    testlog.assert_contains("group 0 optimizer with SGD", "learningRate=0.01,",
                            "velocityScaling=2,")
    testlog.assert_contains("group 1 optimizer with SGD", "learningRate=0.01,",
                            "velocityScaling=2,")

    # Make sure the loss_scaling can be changed, and individual velocityScaling can be set.
    o.loss_scaling = 4.0
    o.param_groups[1]["velocity_scaling"] = 4.0
    o.param_groups[0][
        "loss_scaling"] = 4.0  # doesn't exist: loss scaling is not a group attribute
    model.run(o)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_contains("Ignoring unexpected group 0 attribute",
                            "'loss_scaling'")
    testlog.assert_contains("graph optimizer with SGD",
                            "defaultLearningRate=0.01,",
                            "defaultVelocityScaling=2,", "lossScaling=4")
    testlog.assert_contains("group 0 optimizer with SGD", "learningRate=0.01,",
                            "velocityScaling=2,")
    testlog.assert_contains("group 1 optimizer with SGD", "learningRate=0.01,",
                            "velocityScaling=4,")

    # Make sure the the groups default to the new optimizer's default velocityScaling, manually set lr for both groups
    params = [{
        "params": model.model[0].parameters()
    }, {
        "params": model.model[1].parameters()
    }]
    o = poptorch.optim.SGD(params,
                           lr=0.01,
                           loss_scaling=1.0,
                           velocity_scaling=3.0)
    o.lr = 0.5  # doesn't exit
    o.defaults["lr"] = 0.7
    o.param_groups[0]["lr"] = 0.0
    o.param_groups[1]["lr"] = 1.0
    model.run(o)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_contains("Ignoring unexpected optimizer attribute", "'lr'")
    testlog.assert_contains("graph optimizer with SGD",
                            "defaultLearningRate=0.7,",
                            "defaultVelocityScaling=3,", "lossScaling=1")
    testlog.assert_contains("group 0 optimizer with SGD", "learningRate=0,",
                            "velocityScaling=3,")
    testlog.assert_contains("group 1 optimizer with SGD", "learningRate=1,",
                            "velocityScaling=3,")
