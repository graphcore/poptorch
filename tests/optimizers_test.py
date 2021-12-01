#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import copy
from io import StringIO
import json
import os
import tempfile

import pytest
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import helpers
import poptorch


# Convenience classes for testing
class LAMBNoBias(poptorch.optim.LAMB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, bias_correction=False, **kwargs)


class AdamWNoBias(poptorch.optim.AdamW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, bias_correction=False, **kwargs)


poptorch_optimizers = [
    poptorch.optim.SGD, poptorch.optim.Adam, poptorch.optim.AdamW,
    poptorch.optim.RMSprop, poptorch.optim.LAMB, LAMBNoBias, AdamWNoBias
]

supported_torch_optimizers = [
    optim.SGD, optim.Adam, optim.AdamW, optim.RMSprop
]

all_optimizers = poptorch_optimizers + supported_torch_optimizers


class OptimizerTestModel:
    def __init__(self, num_groups=1, options=None):
        layers = [torch.nn.Linear(10, 10) for _ in range(num_groups)]
        if num_groups == 1:
            self.model = layers[0]
        else:
            self.model = torch.nn.Sequential(*layers)
        self.input = torch.randn(1, 10)
        self.label = torch.randint(0, 10, [1])
        self.poptorch_model = None
        self.options = options

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
                optimizer=optimizer,
                options=self.options)
        elif optimizer:
            self.setOptimizer(optimizer)
        return self.poptorch_model(self.input, self.label)


@pytest.mark.parametrize("opt", all_optimizers)
def test_optimizer(opt):
    torch.manual_seed(42)

    model = OptimizerTestModel()

    # "Train" with learning rate of zero and check the loss remains the same.
    if opt == poptorch.optim.SGD:
        optimizer = opt(model.parameters(), lr=0.00, use_combined_accum=False)
    else:
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
    if opt == poptorch.optim.SGD:
        optimizer = opt(model.parameters(), lr=0.01, use_combined_accum=False)
    else:
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


@helpers.printCapfdOnExit
@pytest.mark.parametrize("opt", (poptorch.optim.Adam, poptorch.optim.AdamW,
                                 AdamWNoBias, poptorch.optim.LAMB, LAMBNoBias))
@pytest.mark.parametrize("accum_type", (torch.float16, torch.float))
@pytest.mark.parametrize("first_order_type", (torch.float16, torch.float))
@pytest.mark.parametrize("second_order_type", (torch.float16, torch.float))
@helpers.overridePoptorchLogLevel("DEBUG")
def test_adam_accum_type(capfd, opt, accum_type, first_order_type,
                         second_order_type):
    def torchTypeToStr(dt):
        t = str(dt)
        assert t in ["torch.float32", "torch.float16"]
        return t.split(".")[1]

    torch.manual_seed(42)
    model = OptimizerTestModel()

    # "Train" with learning rate of zero and check the loss remains the same.
    optimizer = opt(model.parameters(),
                    lr=0.01,
                    accum_type=accum_type,
                    first_order_momentum_accum_type=first_order_type,
                    second_order_momentum_accum_type=second_order_type)
    model.run(optimizer)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_matches(
        "graph optimizer", "accumType=" + torchTypeToStr(accum_type),
        "firstOrderMomentumAccumType=" + torchTypeToStr(first_order_type),
        "secondOrderMomentumAccumType=" + torchTypeToStr(second_order_type))


@helpers.printCapfdOnExit
@pytest.mark.parametrize("accum_type", (torch.float16, torch.float))
@pytest.mark.parametrize("velocity_accum_type", (torch.float16, torch.float))
@helpers.overridePoptorchLogLevel("DEBUG")
def test_sgd_accum_type(capfd, accum_type, velocity_accum_type):
    def torchTypeToStr(dt):
        t = str(dt)
        assert t in ["torch.float32", "torch.float16"]
        return t.split(".")[1]

    torch.manual_seed(42)
    model = OptimizerTestModel()

    # "Train" with learning rate of zero and check the loss remains the same.
    optimizer = poptorch.optim.SGD(model.parameters(),
                                   lr=0.01,
                                   use_combined_accum=False,
                                   accum_type=accum_type,
                                   velocity_accum_type=velocity_accum_type)
    model.run(optimizer)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_matches(
        "graph optimizer", "accumType=" + torchTypeToStr(accum_type),
        "firstOrderMomentumAccumType=" + torchTypeToStr(velocity_accum_type))


@pytest.mark.parametrize("use_combined_accum", (True, False))
def test_velocity_scaling_copy(use_combined_accum):
    torch.manual_seed(42)

    model = OptimizerTestModel()

    # "Train" with learning rate of zero and check the loss remains the same.
    optimizer = poptorch.optim.SGD(
        model.parameters(),
        lr=0.05,
        loss_scaling=0.05,
        velocity_scaling=128.1 if use_combined_accum else None,
        use_combined_accum=use_combined_accum)

    model.run(optimizer)

    # Check copy.copy preserves optimizer PopTorch attributes
    o = copy.copy(optimizer)
    model.setOptimizer(o)
    model.run()


@pytest.mark.parametrize(
    "opt",
    {
        optim.SGD,
        poptorch.optim.SGD  #, optim.Adam, optim.AdamW, optim.RMSprop,
        #poptorch.optim.Adam, poptorch.optim.AdamW, AdamWNoBias,
        #poptorch.optim.RMSprop, poptorch.optim.LAMB, LAMBNoBias
    })
def test_optimizer_groups(opt):
    torch.manual_seed(42)

    model = OptimizerTestModel(num_groups=2)

    # Parameter is a soft copy by default oddly.
    weight1 = model.model[0].weight.clone()
    bias1 = model.model[0].bias.clone()
    weight2 = model.model[1].weight.clone()
    bias2 = model.model[1].bias.clone()

    def get_optims(run_time):

        first_group_lr = 0.0 if run_time == 0 else 0.1
        second_group_lr = 0.1 if run_time == 2 else 0.0

        if opt == poptorch.optim.SGD:
            return opt([{
                'params': model.model[0].parameters(),
                "lr": first_group_lr
            }, {
                'params': model.model[1].parameters(),
                "lr": second_group_lr
            }],
                       lr=0.1,
                       use_combined_accum=False)
        return opt([{
            'params': model.model[0].parameters(),
            "lr": first_group_lr
        }, {
            'params': model.model[1].parameters(),
            "lr": second_group_lr
        }],
                   lr=0.1)

    # Start the optimizer as zero for both groups.
    _, original_loss = model.run(get_optims(run_time=0))

    for _ in range(0, 10):
        out, loss = model.run()

    weight1_post, bias1_post = model.model[0].parameters()
    weight2_post, bias2_post = model.model[1].parameters()

    # Nothing should have changed.
    helpers.assert_allequal(expected=weight1, actual=weight1_post)
    helpers.assert_allequal(expected=weight2, actual=weight2_post)
    helpers.assert_allequal(expected=bias1, actual=bias1_post)
    helpers.assert_allequal(expected=bias2, actual=bias2_post)

    # Check we have not trained the model
    assert loss == original_loss

    # Now update the optimizer to train just one weight
    _, original_loss = model.run(get_optims(1))

    for _ in range(0, 10):
        out, loss = model.run()

    weight1_post, bias1_post = model.model[0].parameters()
    weight2_post, bias2_post = model.model[1].parameters()

    assert loss != original_loss

    assert not torch.equal(weight1, weight1_post)
    helpers.assert_allequal(expected=weight2, actual=weight2_post)
    assert not torch.equal(bias1, bias1_post)
    helpers.assert_allequal(expected=bias2, actual=bias2_post)

    # Now update the optimizer to train just both weight
    _, original_loss = model.run(get_optims(2))

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
    helpers.assert_allequal(expected=weight1, actual=weight1_post)
    helpers.assert_allequal(expected=weight2, actual=weight2_post)
    helpers.assert_allequal(expected=bias1, actual=bias1_post)
    helpers.assert_allequal(expected=bias2, actual=bias2_post)


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
def test_optimizer_SGD_separate_velocity_scale_matched(capfd):
    model = OptimizerTestModel()

    optimizer = poptorch.optim.SGD(model.parameters(),
                                   loss_scaling=2.0,
                                   lr=1.0,
                                   use_combined_accum=False)

    model.run(optimizer)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_contains("lossScaling=2", "defaultVelocityScaling=2")


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


@pytest.mark.parametrize("opt", poptorch_optimizers)
def test_optimizer_const(opt):
    torch.manual_seed(42)

    model = OptimizerTestModel()

    # Initialise the optimiser with the default loss_scaling value
    if opt == poptorch.optim.SGD:
        optimizer = opt(model.parameters(),
                        loss_scaling=1.0,
                        lr=1.0,
                        use_combined_accum=False)
    else:
        optimizer = opt(model.parameters(), loss_scaling=1.0, lr=1.0)

    model.run(optimizer)

    optimizer.loss_scaling = 2.0
    model.run(optimizer)


@pytest.mark.parametrize("opt", poptorch_optimizers)
def test_optimizer_mark_as_variable(opt):
    torch.manual_seed(42)

    model = OptimizerTestModel()
    # Initialise the optimiser with the default loss_scaling value
    if opt == poptorch.optim.SGD:
        optimizer = opt(model.parameters(), lr=1.0, use_combined_accum=False)
    else:
        optimizer = opt(model.parameters(), lr=1.0)

    optimizer.variable_attrs.markAsVariable("loss_scaling")
    model.run(optimizer)

    optimizer.loss_scaling = 2.0
    model.run(optimizer)


@pytest.mark.parametrize("opt", {poptorch.optim.LAMB, LAMBNoBias})
def test_lamb_max_weight_norm(opt):
    torch.manual_seed(42)
    model = OptimizerTestModel()

    optimizer = opt(model.parameters(), lr=0.01, max_weight_norm=100.0)
    _, original_loss = model.run(optimizer)
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
@pytest.mark.parametrize("use_combined_accum", (True, False))
@helpers.overridePoptorchLogLevel("DEBUG")
def test_variable_groups(capfd, use_combined_accum):
    model = OptimizerTestModel(num_groups=2)

    # Make sure all groups have the default values, and the values are not (const)
    params = [{
        "params": model.model[0].parameters()
    }, {
        "params": model.model[1].parameters()
    }]
    o = poptorch.optim.SGD(
        params,
        lr=0.01,
        loss_scaling=2.0,
        velocity_scaling=2.0 if use_combined_accum else None,
        use_combined_accum=use_combined_accum)
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
    o.param_groups[1]["velocity_scaling"] = 4.0  # onl for combined variant
    o.param_groups[0][
        "loss_scaling"] = 4.0  # doesn't exist: loss scaling is not a group attribute
    model.run(o)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_contains("Ignoring unexpected group 0 attribute",
                            "'loss_scaling'")
    if use_combined_accum:
        testlog.assert_contains("graph optimizer with SGD",
                                "defaultLearningRate=0.01,",
                                "defaultVelocityScaling=2,", "lossScaling=4")
        testlog.assert_contains("group 0 optimizer with SGD",
                                "learningRate=0.01,", "velocityScaling=2,")
    else:
        testlog.assert_contains("Ignoring unexpected group 1 attribute",
                                "'velocity_scaling'")
        testlog.assert_contains("group 0 optimizer with SGD",
                                "learningRate=0.01,", "velocityScaling=4,")

    testlog.assert_contains("group 1 optimizer with SGD", "learningRate=0.01,",
                            "velocityScaling=4,")

    # Make sure the the groups default to the new optimizer's default velocityScaling, manually set lr for both groups
    params = [{
        "params": model.model[0].parameters()
    }, {
        "params": model.model[1].parameters()
    }]
    o = poptorch.optim.SGD(
        params,
        lr=0.01,
        loss_scaling=1.0,
        velocity_scaling=3.0 if use_combined_accum else None,
        use_combined_accum=use_combined_accum)
    o.lr = 0.5  # doesn't exit
    o.defaults["lr"] = 0.7
    o.param_groups[0]["lr"] = 0.0
    o.param_groups[1]["lr"] = 1.0
    model.run(o)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_contains("Ignoring unexpected optimizer attribute", "'lr'")

    if use_combined_accum:
        testlog.assert_contains("graph optimizer with SGD",
                                "defaultLearningRate=0.7,",
                                "defaultVelocityScaling=3,", "lossScaling=1")
        testlog.assert_contains("group 0 optimizer with SGD",
                                "learningRate=0,", "velocityScaling=3,")
        testlog.assert_contains("group 1 optimizer with SGD",
                                "learningRate=1,", "velocityScaling=3,")
    else:
        testlog.assert_contains("graph optimizer with SGD",
                                "defaultLearningRate=0.7,",
                                "defaultVelocityScaling=1,", "lossScaling=1")
        testlog.assert_contains("group 0 optimizer with SGD",
                                "learningRate=0,", "velocityScaling=1,")
        testlog.assert_contains("group 1 optimizer with SGD",
                                "learningRate=1,", "velocityScaling=1,")


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
@helpers.overridePoptorchLogLevel("DEBUG")
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
    pytorch_opt = poptorch_opt.__bases__[1]  # Retrieve the upstream type

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
    if poptorch_opt == poptorch.optim.SGD:
        optimizer = poptorch_opt(model.parameters(),
                                 lr=1.0,
                                 use_combined_accum=False)
    else:
        optimizer = poptorch_opt(model.parameters(), lr=1.0)

    model.run(optimizer)
    testlog = helpers.LogChecker(capfd)
    # As the optimizer is identical it shouldn't trigger any update in the backend
    testlog.assert_no_matches("graph optimizer")
    testlog.assert_no_matches("group 0 optimizer")

    # Create a poptorch optimizer and set all the attributes manually: they should all be marked as variable
    # So let's now manually mark them as constant (This should result in the same optimizer as the default one)
    if poptorch_opt == poptorch.optim.SGD:
        optimizer = poptorch_opt(model.parameters(),
                                 lr=1.0,
                                 use_combined_accum=False,
                                 **opt_args)
    else:
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

    if poptorch_opt == poptorch.optim.SGD:
        optimizer = poptorch_opt(model.parameters(),
                                 lr=1.0,
                                 **opt_args,
                                 use_combined_accum=False)
    else:
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

    if poptorch_opt == poptorch.optim.SGD:
        optimizer = poptorch_opt(model.parameters(),
                                 lr=1.0,
                                 use_combined_accum=False,
                                 **new_opts)
    else:
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
    if poptorch_opt == poptorch.optim.SGD:
        optimizer = poptorch_opt(model.parameters(),
                                 lr=1.0,
                                 use_combined_accum=False)
    else:
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


@pytest.mark.parametrize(
    "reduction", (poptorch.ReductionType.Sum, poptorch.ReductionType.Mean))
def test_gradient_accum(reduction):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers = [torch.nn.Linear(10, 10) for _ in range(3)]

            self.model = torch.nn.Sequential(*layers)
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, x, target):
            fwd = self.model(x)
            return fwd, self.loss(fwd, target)

    accum = 20

    opts = poptorch.Options()
    opts.Training.gradientAccumulation(accum)
    opts.Training.accumulationAndReplicationReductionType(reduction)

    model = Model()

    poptorch_model = poptorch.trainingModel(model, options=opts)

    ins = torch.randn([1, 10]).expand(accum, 10)
    target = torch.randint(0, 10, size=[1]).expand(accum)

    _, loss = poptorch_model(ins, target)

    for _ in range(0, 500):
        _, loss = poptorch_model(ins, target)

    # Check we have trained the "model"
    assert loss < 0.03


@pytest.mark.parametrize(
    "reduction", (poptorch.ReductionType.Sum, poptorch.ReductionType.Mean))
def test_gradient_accum_new_api(reduction):
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers = [torch.nn.Linear(10, 10) for _ in range(3)]

            self.model = torch.nn.Sequential(*layers)
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, x, target):
            fwd = self.model(x)
            return fwd, self.loss(fwd, target)

    accum = 20

    opts = poptorch.Options()
    opts.Training.gradientAccumulation(accum)
    opts.Training.accumulationAndReplicationReductionType(reduction)

    model = Model()

    poptorch_model = poptorch.trainingModel(model, options=opts)

    ins = torch.randn([1, 10]).expand(accum, 10)
    target = torch.randint(0, 10, size=[1]).expand(accum)

    _, loss = poptorch_model(ins, target)

    for _ in range(0, 500):
        _, loss = poptorch_model(ins, target)

    # Check we have trained the "model"
    assert loss < 0.03


@helpers.printCapfdOnExit
@pytest.mark.parametrize("use_combined_accum", (True, False))
@helpers.overridePoptorchLogLevel("WARN"
                                  )  # We only want warnings for this test
def test_extra_attributes(capfd, use_combined_accum):
    model = OptimizerTestModel(num_groups=2)

    # Make sure all groups have the default values, and the values are not (const)
    params = [{
        "params": model.model[0].parameters()
    }, {
        "params": model.model[1].parameters()
    }]
    o = poptorch.optim.SGD(
        params,
        lr=0.01,
        loss_scaling=2.0,
        velocity_scaling=2.0 if use_combined_accum else None,
        use_combined_accum=use_combined_accum)
    model.run(o)
    o.step = 0
    o.param_groups[0]["initial_lr"] = 0.1
    o.param_groups[1]["initial_lr"] = 0.1
    model.run(o)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_matches("unexpected optimizer attribute")
    testlog.assert_matches(r"unexpected group \d attribute")
    # loss_scaling = 3.0: Make sure optimizer is different to trigger update
    o.loss_scaling = 3.0
    model.run(o)
    # Ensure warnings are printed only once
    testlog = helpers.LogChecker(capfd)
    testlog.assert_no_matches("unexpected optimizer attribute")
    testlog.assert_no_matches(r"unexpected group \d attribute")


@helpers.printCapfdOnExit
@pytest.mark.parametrize("use_combined_accum", (True, False))
@helpers.overridePoptorchLogLevel("WARN"
                                  )  # We only want warnings for this test
def test_extra_attributes2(capfd, use_combined_accum):

    opts = poptorch.Options()
    opts.relaxOptimizerAttributesChecks()
    model = OptimizerTestModel(num_groups=2, options=opts)
    # Make sure all groups have the default values, and the values are not (const)
    params = [{
        "params": model.model[0].parameters()
    }, {
        "params": model.model[1].parameters()
    }]
    o = poptorch.optim.SGD(
        params,
        lr=0.01,
        loss_scaling=2.0,
        velocity_scaling=2.0 if use_combined_accum else None,
        use_combined_accum=use_combined_accum)
    model.run(o)
    o.step = 0
    o.param_groups[0]["initial_lr"] = 0.1
    o.param_groups[1]["initial_lr"] = 0.1
    model.run(o)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_no_matches("unexpected optimizer attribute")
    testlog.assert_no_matches(r"unexpected group \d attribute")


@helpers.printCapfdOnExit
@pytest.mark.parametrize("use_combined_accum", (True, False))
@helpers.overridePoptorchLogLevel("WARN"
                                  )  # We only want warnings for this test
def test_extra_attributes3(capfd, use_combined_accum):
    model = OptimizerTestModel(num_groups=2)
    # Make sure all groups have the default values, and the values are not (const)
    params = [{
        "params": model.model[0].parameters()
    }, {
        "params": model.model[1].parameters()
    }]
    o = poptorch.optim.SGD(
        params,
        lr=0.01,
        loss_scaling=2.0,
        velocity_scaling=2.0 if use_combined_accum else None,
        use_combined_accum=use_combined_accum)
    o.step = 0
    o.param_groups[0]["initial_lr"] = 0.1
    o.param_groups[1]["initial_lr"] = 0.1
    model.run(o)
    # If extra attributes are added before the first run
    # they shouldn't trigger any warning
    testlog = helpers.LogChecker(capfd)
    testlog.assert_no_matches("unexpected optimizer attribute")
    testlog.assert_no_matches(r"unexpected group \d attribute")

    # loss_scaling = 4.0: Make sure optimizer is different to trigger update
    o.loss_scaling = 4.0
    # initial_lr is a group attribute: should trigger a warning.
    o.initial_lr = 0.2
    # If they're added later then they should print a warning
    model.run(o)
    testlog = helpers.LogChecker(capfd)
    testlog.assert_matches("unexpected optimizer attribute")
    testlog.assert_no_matches(r"unexpected group \d attribute")


@pytest.mark.parametrize("use_tf_variant", [True, False])
def test_rmsprop_tf_variant(use_tf_variant):
    torch.manual_seed(0)
    # Make sure the TF flag is propagated correctly by comparing the
    # results of TF and non-TF versions.
    weight = torch.randn(10, 10)
    bias = torch.randn(10)
    input = torch.randn(1, 10)
    label = torch.randint(0, 10, [1])

    model_pt = OptimizerTestModel()
    model_pt.model.weight = torch.nn.Parameter(weight.detach().clone())
    model_pt.model.bias = torch.nn.Parameter(bias.detach().clone())
    model_pt.input = input.detach().clone()
    model_pt.label = label.detach().clone()
    optimizer_pt = poptorch.optim.RMSprop(model_pt.parameters(), lr=0.02)

    model_tf = OptimizerTestModel()
    model_tf.model.weight = torch.nn.Parameter(weight.detach().clone())
    model_tf.model.bias = torch.nn.Parameter(bias.detach().clone())
    model_tf.input = input.detach().clone()
    model_tf.label = label.detach().clone()
    optimizer_tf = poptorch.optim.RMSprop(model_tf.parameters(),
                                          lr=0.02,
                                          use_tf_variant=use_tf_variant)

    helpers.assert_allequal(actual=model_pt.model.weight,
                            expected=model_tf.model.weight)
    helpers.assert_allequal(actual=model_pt.model.bias,
                            expected=model_tf.model.bias)

    for _ in range(5):
        out_pt, loss_pt = model_pt.run(optimizer_pt)
        out_tf, loss_tf = model_tf.run(optimizer_tf)

    if use_tf_variant:
        assert not torch.allclose(model_pt.model.weight, model_tf.model.weight)
        assert not torch.allclose(out_pt, out_tf)
        assert not torch.allclose(loss_pt, loss_tf)
    else:
        helpers.assert_allequal(
            actual=model_pt.model.weight.detach().clone(),
            expected=model_tf.model.weight.detach().clone())
        helpers.assert_allequal(actual=out_pt, expected=out_tf)
        helpers.assert_allequal(actual=loss_pt, expected=loss_tf)


@pytest.mark.parametrize("opt", all_optimizers)
def test_optimizer_results(opt):
    torch.manual_seed(42)

    class Stepper:
        def __init__(self, model, lr, optimizer):
            self.lr = lr
            self.setup_cpu(model, optimizer)
            self.setup_ipu(model, optimizer)
            self.check_parameters()

        def setup_cpu(self, model, optimizer):
            self.cpu_model = copy.deepcopy(model)
            self.optimizer = optimizer(self.cpu_model.parameters(), lr=self.lr)

        def setup_ipu(self, model, optimizer):
            self.ipu_model = copy.deepcopy(model)
            ipu_optimizer = optimizer(self.ipu_model.parameters(), lr=self.lr)
            self.training_model = poptorch.trainingModel(
                self.ipu_model, optimizer=ipu_optimizer)

        def check_parameters(self):
            for cpu, ipu in zip(self.cpu_model.named_parameters(),
                                self.ipu_model.named_parameters()):
                cpu = cpu[1]
                ipu = ipu[1]
                helpers.assert_allclose(actual=ipu, expected=cpu)

        def cpu_step(self, batch):
            self.optimizer.zero_grad()
            _, loss = self.cpu_model(batch)
            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            return loss

        def ipu_step(self, batch):
            _, loss = self.training_model(batch)
            return loss

    num_samples = 10
    X = torch.rand(num_samples)
    lr = 0.01
    num_steps = 10

    cpu_loss = torch.empty(num_steps)
    ipu_loss = torch.empty(num_steps)

    stepper = Stepper(helpers.ModelWithWeights(torch.nn.LogSoftmax(), X.shape),
                      lr=lr,
                      optimizer=opt)

    for i in range(num_steps):
        cpu_loss[i] = stepper.cpu_step((X, ))
        ipu_loss[i] = stepper.ipu_step((X, ))

        stepper.check_parameters()

    helpers.assert_allclose(expected=cpu_loss,
                            actual=ipu_loss,
                            atol=1e-5,
                            rtol=1e-5)


@pytest.mark.parametrize("opt", [(optim.SGD, poptorch.optim.SGD),
                                 (optim.Adam, poptorch.optim.Adam),
                                 (optim.AdamW, poptorch.optim.AdamW)],
                         ids=['SGD', 'Adam', 'AdamW'])
def test_gradient_clipping(opt):
    torch.manual_seed(42)
    max_norm = 0.001

    class Stepper:
        def __init__(self, model, lr, optimizer):
            self.lr = lr
            self.original_model = model
            self.setup_torch(model, optimizer[0])
            self.setup_poptorch(model, optimizer[1])
            self.check_parameters()

        def setup_torch(self, model, optimizer):
            self.torch_model = copy.deepcopy(model)
            self.optimizer = optimizer(self.torch_model.parameters(),
                                       lr=self.lr)

        def setup_poptorch(self, model, optimizer):
            self.ipu_model = copy.deepcopy(model)
            ipu_optimizer = optimizer(self.ipu_model.parameters(),
                                      lr=self.lr,
                                      max_grad_norm=max_norm)
            self.training_model = poptorch.trainingModel(
                self.ipu_model, optimizer=ipu_optimizer)

        def check_parameters(self):
            for expected, actual in zip(
                    self.torch_model.named_parameters(),
                    self.training_model.named_parameters()):
                expected = expected[1]
                actual = actual[1]
                helpers.assert_allclose(actual=actual,
                                        expected=expected,
                                        atol=1e-5,
                                        rtol=1e-5)

        def torch_step(self, batch):
            self.optimizer.zero_grad()
            _, loss = self.torch_model(batch)
            loss = loss.sum()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.torch_model.parameters(),
                                           max_norm)

            self.optimizer.step()
            return loss

        def poptorch_step(self, batch):
            _, loss = self.training_model(batch)
            return loss

    num_samples = 10
    X = torch.randn(num_samples)
    lr = 0.01
    num_steps = 10

    torch_loss = torch.empty(num_steps)
    poptorch_loss = torch.empty(num_steps)

    stepper = Stepper(helpers.ModelWithWeights(torch.nn.LogSoftmax(), X.shape),
                      lr=lr,
                      optimizer=opt)

    for i in range(num_steps):
        torch_loss[i] = stepper.torch_step((X, ))
        poptorch_loss[i] = stepper.poptorch_step((X, ))

        stepper.check_parameters()

    helpers.assert_allclose(expected=torch_loss,
                            actual=poptorch_loss,
                            atol=1e-5,
                            rtol=1e-5)


@pytest.mark.parametrize("optim", poptorch_optimizers)
def test_read_ipu_state(optim):
    torch.manual_seed(42)
    input = torch.randn(3)
    # A simple model with weights and a loss function
    model = helpers.ModelWithWeights(lambda x: x, input.shape)

    lr = 0.05
    wd = 0.025
    ls = 0.75

    optimizer = optim(model.parameters(),
                      lr=lr,
                      weight_decay=wd,
                      loss_scaling=ls)
    training_model = poptorch.trainingModel(model, optimizer=optimizer)

    # Before the model is compiled, the state_dict should throw an error
    with pytest.raises(
            RuntimeError,
            match="You must compile or run a training model with this "
            "optimizer before the state can be read."):
        optimizer.state_dict()

    # Compiling should populate the state_dict
    training_model.compile((input, ))
    s0 = optimizer.state_dict()

    sgd_param_keys = [
        "scaledLearningRate0___specific___model.lin.bias",
        "scaledLearningRate0___specific___model.lin.weight",
        "weightDecayScaleFactor0___specific___model.lin.bias",
        "weightDecayScaleFactor0___specific___model.lin.weight"
    ]
    non_sgd_param_keys = [
        "learningRate___specific___model.lin.bias",
        "learningRate___specific___model.lin.weight",
        "weightDecay___specific___model.lin.bias",
        "weightDecay___specific___model.lin.weight"
    ]

    # Check that shared keys are present and user provided values are read
    # back correctly
    if isinstance(optimizer, torch.optim.SGD):
        for k in sgd_param_keys:
            assert k in s0["ipu_param"].keys()

        # weightDecayScaleFactor0 =
        # 1 - lr * (1 - dm) * wd, dm = 0
        wdsf0 = 1 - lr * wd
        helpers.assert_allclose(
            actual=s0["ipu_param"]
            ["weightDecayScaleFactor0___specific___model.lin.bias"],
            expected=torch.tensor(wdsf0))

        # scaledLearningRate0 =
        # lr *  (1 - dm) / ls, dm = 0
        slr0 = lr / ls
        helpers.assert_allclose(
            actual=s0["ipu_param"]
            ["scaledLearningRate0___specific___model.lin.bias"],
            expected=torch.tensor(slr0))
    else:
        # Only non-SGD optimisers have state tensors
        state_keys = ["Accl1___model.lin.weight", "Accl1___model.lin.bias"]
        for k in non_sgd_param_keys:
            assert k in s0["ipu_param"].keys()
        for k in state_keys:
            assert k in s0["ipu_state"].keys()

        helpers.assert_allclose(
            actual=s0["ipu_param"]["learningRate___specific___model.lin.bias"],
            expected=torch.tensor(lr))
        helpers.assert_allclose(
            actual=s0["ipu_param"]["weightDecay___specific___model.lin.bias"],
            expected=torch.tensor(wd))

        # Run the model, get the updated state dict and check optimiser state tensors have changed
        training_model((input, ))
        s1 = optimizer.state_dict()
        assert not all([
            torch.equal(s0["ipu_state"][k], s1["ipu_state"][k])
            for k in s0["ipu_state"].keys()
        ])

    helpers.assert_allclose(actual=s0["ipu_param"]["lossScaling_FLOAT"],
                            expected=torch.tensor(ls))


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
def test_read_ipu_state_cached(caplog, capfd):
    input = torch.ones(3)
    # A simple model with weights and a loss function
    model = helpers.ModelWithWeights(lambda x: x, input.shape)
    optimizer = poptorch.optim.SGD(model.parameters(), lr=0.0)

    training_model = poptorch.trainingModel(model, optimizer=optimizer)

    training_model.compile((input, ))
    # Compilation should trigger an optimiser state IPU->host copy
    optimizer.state_dict()
    log = helpers.LogChecker(capfd)
    log.assert_matches("Writing optimiser state tensors from IPU to host.")

    # The second invocation should use the cached state dict, since
    # the internal optimiser state hasn't changed
    optimizer.state_dict()
    assert "Using cached optimiser state dict" in caplog.text


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.parametrize("optim", [poptorch.optim.SGD, torch.optim.SGD])
def test_read_ipu_state_on_detach(caplog, capfd, optim):
    input = torch.ones(3)
    # A simple model with weights and a loss function
    model = helpers.ModelWithWeights(lambda x: x, input.shape)
    optimizer = optim(model.parameters(), lr=0.0)

    training_model = poptorch.trainingModel(model, optimizer=optimizer)

    training_model.compile((input, ))
    training_model.detachFromDevice()

    # Detach should trigger an optimiser state IPU->host copy for PopTorch optimizers
    log = helpers.LogChecker(capfd)
    if isinstance(optimizer, poptorch.optim.Optimizer):
        log.assert_matches("Writing optimiser state tensors from IPU to host.")
    else:
        log.assert_no_matches(
            "Writing optimiser state tensors from IPU to host.")

    # The second invocation should use the cached state dict, since
    # the internal optimiser state hasn't changed
    state = optimizer.state_dict()
    log = helpers.LogChecker(caplog.text)
    if isinstance(optimizer, poptorch.optim.Optimizer):
        log.assert_matches("Using cached optimiser state dict")
    else:
        log.assert_no_matches("Using cached optimiser state dict")

    optimizer.load_state_dict(state)

    training_model.attachToDevice()
    # Detach should trigger an optimiser state IPU->host copy for PopTorch optimizers
    log = helpers.LogChecker(capfd)
    if isinstance(optimizer, poptorch.optim.Optimizer):
        log.assert_matches(
            "Writing optimiser state tensors from host to IPU memory")
    else:
        log.assert_no_matches(
            "Writing optimiser state tensors from host to IPU memory")


@pytest.mark.parametrize("optim", poptorch_optimizers)
def test_write_ipu_state(optim):
    torch.manual_seed(42)
    input = torch.randn(3)
    # A simple model with weights and a loss function
    model = helpers.ModelWithWeights(lambda x: x, input.shape)

    # SGD requires LR to be specified but the value doesn't matter
    optimizer = optim(model.parameters(), lr=0.0)
    training_model = poptorch.trainingModel(model, optimizer=optimizer)

    # Compiling should populate the state_dict
    training_model.compile((input, ))
    # The initial optimiser state
    s0 = optimizer.state_dict()

    # Just set values randomly so we can check they changed
    for k, v in s0["ipu_param"].items():
        s0["ipu_param"][k] = torch.randn_like(v)
    if not isinstance(optimizer, torch.optim.SGD):
        # Only non-SGD optimisers have state tensors
        for k, v in s0["ipu_state"].items():
            s0["ipu_state"][k] = torch.randn_like(v)

    # Load the modified state dict
    optimizer.load_state_dict(s0)

    # Read it back into a new dict
    s1 = optimizer.state_dict()

    # Check that the values read back match the ones set
    for k, v in s0["ipu_param"].items():
        helpers.assert_allequal(actual=s1["ipu_param"][k], expected=v)
    if not isinstance(optimizer, torch.optim.SGD):
        # Only non-SGD optimisers have state tensors
        for k, v in s0["ipu_state"].items():
            helpers.assert_allequal(actual=s1["ipu_state"][k], expected=v)


def test_write_ipu_state_from_cpu():
    input = torch.ones(2)
    lin = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(lin.parameters())

    # Perform a CPU training step to populate torch optimiser state
    out = lin(input)
    out.backward()
    optimizer.step()

    pop_optimizer = poptorch.optim.Adam(lin.parameters())
    # Try to load the CPU optimiser state onto the IPU
    with pytest.raises(
            RuntimeError,
            match="Only IPU optimizer states can be loaded onto the IPU."):
        pop_optimizer.load_state_dict(optimizer.state_dict())


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
def test_write_ipu_state_before_override(capfd):
    input = torch.ones(2)
    model = helpers.ModelWithWeights(lambda x: x, input.shape)
    optimizer = poptorch.optim.Adam(model.parameters())
    training_model = poptorch.trainingModel(model, optimizer=optimizer)

    # Compile and run the model, get the state dict
    training_model((input, ))
    s1 = optimizer.state_dict()

    # Create a new optimiser and load the state dict
    new_optimizer = poptorch.optim.Adam(model.parameters())
    new_optimizer.load_state_dict(s1)

    # Compile a new model with the new loaded optimiser
    new_training_model = poptorch.trainingModel(model, optimizer=new_optimizer)
    new_training_model.compile((input, ))

    # The state read back should match the initial state
    s2 = new_optimizer.state_dict()
    for k, v in s1["ipu_state"].items():
        helpers.assert_allequal(actual=s2["ipu_state"][k], expected=v)

    # Confirm that an optimiser state IPU->host copy actually took place
    log = helpers.LogChecker(capfd)
    log.assert_matches("Writing optimiser state tensors from IPU to host.")


def test_write_ipu_state_from_checkpoint():
    input = torch.ones(2)
    model = helpers.ModelWithWeights(lambda x: x, input.shape)
    optimizer = poptorch.optim.Adam(model.parameters(), lr=1.0)
    # Halve the LR after each training step
    scheduler = ExponentialLR(optimizer, 0.5)
    training_model = poptorch.trainingModel(model, optimizer=optimizer)

    # Compile and run the model
    training_model((input, ))
    # Step the scheduler for the next epoch
    # lr: 1.0 -> 0.5
    scheduler.step()
    # Set the new LR
    training_model.setOptimizer(optimizer)
    s1 = optimizer.state_dict()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "checkpoint.pt")
        # Save the state_dict to file
        torch.save({"optimizer_state_dict": s1}, path)
        # Load it back
        checkpoint = torch.load(path)

        # Create a new optimizer and load the checkpoint
        optimizer = poptorch.optim.Adam(model.parameters(), lr=0.1)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        s2 = optimizer.state_dict()
        # Ensure the new optimizer state matches the one saved
        for k, v in s1["ipu_state"].items():
            helpers.assert_allequal(actual=s2["ipu_state"][k], expected=v)
        for k, v in s1["ipu_param"].items():
            helpers.assert_allequal(actual=s2["ipu_param"][k], expected=v)

        # Now continue training to test that the updated LR is used
        scheduler = ExponentialLR(optimizer, 0.5)
        training_model.setOptimizer(optimizer)
        # New LR is set internally when the model is run
        training_model((input, ))
        s3 = optimizer.state_dict()
        torch_lr = torch.tensor(s3["param_groups"][0]["lr"])
        poptorch_lr = s3["ipu_param"][
            "learningRate___specific___model.lin.bias"]
        # Ensure the torch LR parameter is correct
        helpers.assert_allclose(actual=torch_lr, expected=torch.tensor(0.5))
        # Ensure the internal LR parameter matches
        helpers.assert_allclose(actual=poptorch_lr, expected=torch_lr)


def test_setOptimizer_frozen_options_ok():
    input = torch.ones(2)
    opts = poptorch.Options()
    opts.Training.setMeanAccumulationAndReplicationReductionStrategy(
        poptorch.MeanReductionStrategy.Post)

    model = helpers.ModelWithWeights(lambda x: x, input.shape)
    # This will freeze the options
    data = poptorch.DataLoader(opts, [(torch.ones(2), )])

    optimizer = poptorch.optim.Adam(model.parameters(),
                                    lr=0.5,
                                    accum_type=torch.half)

    # will set the reduction strategy to Running
    training_model = poptorch.trainingModel(model,
                                            optimizer=optimizer,
                                            options=opts)
    training_model.compile(next(iter(data)))
    assert training_model.options.Training.meanAccumulationAndReplicationReductionStrategy == poptorch.MeanReductionStrategy.Running  # pylint: disable=line-too-long

    optimizer.param_groups[0]['lr'] = 0.01
    training_model.setOptimizer(optimizer)


def test_setOptimizer_frozen_options_broken():
    input = torch.ones(2)
    opts = poptorch.Options()
    opts.Training.setMeanAccumulationAndReplicationReductionStrategy(
        poptorch.MeanReductionStrategy.Post)

    model = helpers.ModelWithWeights(lambda x: x, input.shape)
    # This will freeze the options
    data = poptorch.DataLoader(opts, [(torch.ones(2), )])

    optimizer = poptorch.optim.Adam(model.parameters(), lr=0.5)

    # will set the reduction strategy to Running
    training_model = poptorch.trainingModel(model,
                                            optimizer=optimizer,
                                            options=opts)
    training_model.compile(next(iter(data)))
    assert training_model.options.Training.meanAccumulationAndReplicationReductionStrategy == poptorch.MeanReductionStrategy.Post  # pylint: disable=line-too-long

    optimizer.param_groups[0]['lr'] = 0.01
    optimizer.accum_type = torch.half
    with pytest.raises(ValueError, match="is already compiled"):
        training_model.setOptimizer(optimizer)
