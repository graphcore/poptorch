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


@helpers.printCapfdOnExit
@pytest.mark.parametrize("opt", (
    (poptorch.optim.SGD, (("momentum", 0.0), ("dampening", 0.0),
                          ("weight_decay", 0.0))),
    (poptorch.optim.Adam, (("betas", (0.9, 0.999)), ("eps", 1e-08),
                           ("weight_decay", 0.0), ("amsgrad", False))),
    (poptorch.optim.AdamW, (("betas", (0.9, 0.999)), ("eps", 1e-08),
                            ("weight_decay", 0.01), ("amsgrad", False))),
    (poptorch.optim.RMSprop, (("momentum", 0.0), ("alpha", 0.99),
                              ("eps", 1e-08), ("weight_decay", 0.0))),
))
# pylint: disable=too-many-statements
def test_variable_default(opt, capfd):
    def toCamelCase(string):
        """Convert a snake case string (Pytorch) to camel case (Popart)"""
        words = string.split("_")
        return words[0] + "".join(w.capitalize() for w in words[1:])

    def toPopartName(name, default):
        if name == "lr":
            name = "learning_rate"
        # amsgrad doesn't get passed to the backend
        if name in ["amsgrad"]:
            return []
        if name == "betas":
            return toPopartName("beta1", default) + toPopartName(
                "beta2", default)
        if default:
            name = "default_" + name
        return [toCamelCase(name)]

    def createExpr(attr, is_const=True):
        const_expr = r" \(const\)"
        if not is_const:
            const_expr = "(?!" + const_expr + ")"

        return r"%s=[^ ,]+%s" % (attr, const_expr)

    def genRegexp(attrs, default=False, is_const=False):
        if isinstance(attrs, str):
            attrs = [attrs]
        exprs = []
        for a in attrs:
            for n in toPopartName(a, default):
                exprs.append(createExpr(n, is_const))
        return exprs

    # All the attribute values in "opt" are the default pytorch values which
    # means if the user instantiate a pytorch optimizer with them, we'll
    # consider all these attributes as constant.
    # However if a poptorch optimizer is used then they will all be considered
    # as variable because they were explicitly passed to the constructor.
    poptorch_opt, opt_args_tuple = opt
    opt_args = dict(opt_args_tuple)
    poptorch.setLogLevel(1)  # Force debug logging
    pytorch_opt = poptorch_opt.__bases__[0]  # Retrieve the upstream type

    # Learning rate is a special case: it's always variable so handle it separately.
    attrs = list(opt_args.keys())

    # Test the torch Optimizer: check all the attributes are set to constant by default
    model = OptimizerTestModel()
    optimizer = pytorch_opt(model.parameters(), lr=1.0, **opt_args)
    model.run(optimizer)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_matches("graph optimizer",
                           *genRegexp(attrs, default=True, is_const=True),
                           *genRegexp("lr", default=True, is_const=False))
    testlog.assert_matches("group 0 optimizer",
                           *genRegexp(attrs, is_const=True),
                           *genRegexp("lr", is_const=False))

    # Create a default pytorch optimizer (It should be identical to the previous one)
    optimizer = pytorch_opt(model.parameters(), lr=1.0)
    model.run(optimizer)
    testlog = helpers.LogChecker(capfd)
    # As the optimizer is identical it shouldn't trigger any update in the backend
    testlog.assert_no_matches("graph optimizer")
    testlog.assert_no_matches("group 0 optimizer")

    # Create a default poptorch optimizer (As we don't explicitly specify any attribute they will all be considered as constant)
    optimizer = poptorch_opt(model.parameters(), lr=1.0)
    model.run(optimizer)
    testlog = helpers.LogChecker(capfd)
    # As the optimizer is identical it shouldn't trigger any update in the backend
    testlog.assert_no_matches("graph optimizer")
    testlog.assert_no_matches("group 0 optimizer")

    # Create a poptorch optimizer and set all the attributes manually: they should all be marked as variable
    # So let's now manually mark them as constant (This should result in the same optimizer as the default one)
    optimizer = poptorch_opt(model.parameters(), lr=1.0, **opt_args)
    for attr in opt_args.keys():
        assert not optimizer.variable_attrs.isConstant(attr)
        optimizer.variable_attrs.markAsConstant(attr)
    model.run(optimizer)
    # As the optimizer is identical it shouldn't trigger any update in the backend
    testlog.assert_no_matches("graph optimizer")
    testlog.assert_no_matches("group 0 optimizer")

    # Test the poptorch Optimizer: check all the manually set attributes are set to variable by default
    # Create a new model as the optimizers would otherwise mismatch
    model = OptimizerTestModel()
    optimizer = poptorch_opt(model.parameters(), lr=1.0, **opt_args)
    model.run(optimizer)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_matches("graph optimizer",
                           *genRegexp(attrs, default=True, is_const=False),
                           *genRegexp("lr", default=True, is_const=False))
    testlog.assert_matches("group 0 optimizer",
                           *genRegexp(attrs, is_const=False),
                           *genRegexp("lr", is_const=False))

    # Check the values can actually change
    new_opts = {}
    for k, v in opt_args.items():
        if isinstance(v, float):
            new_opts[k] = v + 0.5
        elif isinstance(v, tuple):
            new_opts[k] = tuple(elt / 2.0 for elt in v)
        else:
            new_opts[k] = v
    optimizer = poptorch_opt(model.parameters(), lr=1.0, **new_opts)
    model.run(optimizer)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_matches("graph optimizer",
                           *genRegexp(attrs, default=True, is_const=False),
                           *genRegexp("lr", default=True, is_const=False))
    testlog.assert_matches("group 0 optimizer",
                           *genRegexp(attrs, is_const=False),
                           *genRegexp("lr", is_const=False))

    # Check we can manually mark attributes as variable
    optimizer = poptorch_opt(model.parameters(), lr=1.0)
    for attr in opt_args.keys():
        assert optimizer.variable_attrs.isConstant(attr)
        optimizer.variable_attrs.markAsVariable(attr)
    model.run(optimizer)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_matches("graph optimizer",
                           *genRegexp(attrs, default=True, is_const=False),
                           *genRegexp("lr", default=True, is_const=False))
    testlog.assert_matches("group 0 optimizer",
                           *genRegexp(attrs, is_const=False),
                           *genRegexp("lr", is_const=False))
