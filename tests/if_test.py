#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from functools import lru_cache
import torch
import pytest
from helpers import assert_allclose
import poptorch


@lru_cache
def infer_model(model):
    return poptorch.inferenceModel(model)


def if_else_harness(model, expected_then, expected_else, *args):
    inference_model = infer_model(model)

    condition = torch.tensor([True])
    ipu_result = inference_model(condition, *args)
    cpu_result = model(condition, *args)
    assert_allclose(expected=expected_then, actual=cpu_result)
    assert_allclose(expected=expected_then, actual=ipu_result)

    condition = torch.tensor([False])
    ipu_result = inference_model(condition, *args)
    cpu_result = model(condition, *args)
    assert_allclose(expected=expected_else, actual=cpu_result)
    assert_allclose(expected=expected_else, actual=ipu_result)


@pytest.mark.skip(
    reason="Returning constant from model does not work in poptorch (AFS-251)")
def test_constants():
    class Model(torch.nn.Module):
        def forward(self, condition):
            def body_then():
                return torch.tensor([0])

            def body_else():
                return torch.tensor([1])

            return poptorch.cond(condition, body_then, [], body_else, [])[0]

    args = [torch.tensor([v]) for v in range(2)]
    if_else_harness(Model(), args[0], args[1])


@pytest.mark.skip(reason="Inplace op does not update model input (AFS-252)")
def test_inplace_op():
    class Model(torch.nn.Module):
        def forward(self, condition, x, y):
            def body_then(a):
                return a.add_(a)

            def body_else(b):
                return b

            return poptorch.cond(condition, body_then, [x], body_else, [y])[0]

    or_x = 1.
    x = torch.tensor([or_x])
    y = torch.tensor([10.])
    exp_then = x + y
    exp_else = y
    if_else_harness(Model(), exp_then, exp_else, x, y)
    assert torch.tensor([or_x]) == x


def test_operations_on_constants():
    constants = [[1., 2.], [3., 4.]]

    class Model(torch.nn.Module):
        def forward(self, condition):
            x = torch.tensor(constants[0])
            y = torch.tensor(constants[1])

            def body_then(a, b):
                a = a * 2
                b = a * b
                return b

            def body_else(a, b):
                a = a - 2
                b = b + a
                return b

            return poptorch.cond(condition, body_then, [x, y], body_else,
                                 [x, y])[0]

    args = []
    exp_then = torch.tensor(
        [a * 2 * b for a, b in zip(constants[0], constants[1])])
    exp_else = torch.tensor(
        [a - 2 + b for a, b in zip(constants[0], constants[1])])
    if_else_harness(Model(), exp_then, exp_else, *args)


def test_body_args():
    class Model(torch.nn.Module):
        def forward(self, condition, x, y):
            def body_then(a):
                out = a + a
                out = out + out
                return out

            def body_else(b):
                return b

            return poptorch.cond(condition, body_then, [x], body_else, [y])[0]

    args = [torch.rand(1) for _ in range(2)]
    if_else_harness(Model(), args[0] * 4, args[1], *args)


def test_cond_training():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(4, 4)

        def forward(self, condition, x):
            def body(x):
                return self.layer1(x)

            out = poptorch.cond(condition, body, [x], body, [x])[0]
            loss = poptorch.identity_loss(out, reduction='sum')
            return out, loss

    training_model = poptorch.trainingModel(Model())

    condition = torch.tensor([True])
    x = torch.ones(1, 4).to(torch.float)
    with pytest.raises(
            poptorch.Error,
            match=r"poptorch.cond\(\) is only supported in inference"):
        training_model(condition, x)


def test_multi_outs():
    class Model(torch.nn.Module):
        def forward(self, condition, x, y):
            def body_then(a):
                out1 = x + y
                return a, out1, y

            def body_else(b):
                return b * y, y, x - y

            return poptorch.cond(condition, body_then, [x], body_else, [y])

    args = [torch.rand(1) for _ in range(2)]
    exp_then = [args[0], args[0] + args[1], args[1]]
    exp_else = [args[1] * args[1], args[1], args[0] - args[1]]
    if_else_harness(Model(), exp_then, exp_else, *args)


def test_diff_num_of_args():
    class Model(torch.nn.Module):
        def forward(self, condition, x, y):
            def body_then(x, y):
                return x + y

            def body_else(x):
                return x

            return poptorch.cond(condition, body_then, [x, y], body_else,
                                 [x])[0]

    args = [torch.rand(1) for v in range(2)]
    exp_then = args[0] + args[1]
    if_else_harness(Model(), exp_then, args[0], *args)


def test_args_from_main_graph():
    class Model(torch.nn.Module):
        def forward(self, condition, x, y):
            def body_then():
                return x * y

            def body_else():
                return x

            return poptorch.cond(condition, body_then, [], body_else, [])[0]

    args = [torch.rand(1) for v in range(2)]
    exp_then = args[0] * args[1]
    if_else_harness(Model(), exp_then, args[0], *args)


def test_call_outer_body():
    class Model(torch.nn.Module):
        def forward(self, condition, x, y):
            def outer_body():
                return x + y

            def body_then():
                return outer_body()

            def body_else():
                return x

            return poptorch.cond(condition, body_then, [], body_else, [])[0]

    args = [torch.rand(1) for v in range(2)]
    exp_then = args[0] + args[1]
    if_else_harness(Model(), exp_then, args[0], *args)


def test_args_internal():
    internal_inps = [[10., -10.], [0, -2]]

    class Model(torch.nn.Module):
        def forward(self, *args):
            condition = args[0]
            x = args[1]

            def body_then(a, b):
                return a + b

            def body_else(a):
                return a + x

            in1 = torch.tensor(internal_inps[0])
            return poptorch.cond(condition, body_then,
                                 [in1, torch.tensor(internal_inps[1])],
                                 body_else, [in1])[0]

    input_val = [5., -1.]
    args = [torch.tensor(input_val)]
    exp_then = torch.tensor(internal_inps[0]) + torch.tensor(internal_inps[1])
    exp_else = torch.tensor(internal_inps[0]) + args[0]
    if_else_harness(Model(), exp_then, exp_else, *args)


def test_single_body():
    class Model(torch.nn.Module):
        def forward(self, condition, x, y):
            def body(a, b):
                return a + b

            return poptorch.cond(condition, body, [x, y], body, [x, x])[0]

    args = [torch.rand(1) for _ in range(2)]
    exp_then = torch.tensor(args[0] + args[1])
    exp_else = torch.tensor(args[0] + args[0])
    if_else_harness(Model(), exp_then, exp_else, *args)


def test_nested_cond():
    class Model(torch.nn.Module):
        def forward(self, condition, cond_nested, x, y):
            def body_then():
                def nested_then(x, y):
                    return x + y

                def nested_else():
                    return x - y

                return poptorch.cond(cond_nested, nested_then, [x, y],
                                     nested_else, [])[0]

            def body_else(cond_nested):
                cond_nested = torch.logical_not(cond_nested)

                def nested_then(y):
                    return x * y

                def nested_else():
                    return x * 2

                return poptorch.cond(cond_nested, nested_then, [y],
                                     nested_else, [])[0]

            return poptorch.cond(condition, body_then, [], body_else,
                                 [cond_nested])[0]

    model = Model()
    cond_nested = torch.tensor([True])
    args = [cond_nested] + [torch.rand(1) for v in range(2)]
    exp_then = args[1] + args[2]
    exp_else = args[1] * 2
    if_else_harness(model, exp_then, exp_else, *args)

    cond_nested = torch.tensor([False])
    args = [cond_nested] + [torch.rand(1) for v in range(2)]
    exp_then = args[1] - args[2]
    exp_else = args[1] * args[2]
    if_else_harness(model, exp_then, exp_else, *args)


@pytest.mark.parametrize(
    ("execution_strategy"),
    [
        poptorch.ShardedExecution,
        poptorch.ParallelPhasedExecution,
        poptorch.SerialPhasedExecution,
    ],
)
def test_if_on_multiple_ipus(execution_strategy):
    class Model(torch.nn.Module):
        def forward(self, condition, x, y):
            def body_then(x, y):
                return x + y, y

            def body_else(x, y):
                return x, x * y

            with poptorch.Block("0", ipu_id=0):
                x, y = poptorch.cond(condition, body_then, [x, y], body_else,
                                     [x, y])

            with poptorch.Block("1", ipu_id=1):
                x, y = poptorch.cond(torch.logical_not(condition), body_then,
                                     [x, y], body_else, [x, y])
            return x, y

    stages = [poptorch.Stage(f"{k}") for k in range(0, 2)]
    strategy = execution_strategy(*stages)

    opts = poptorch.Options()
    opts.autoRoundNumIPUs(True)
    opts.setExecutionStrategy(strategy)
    ipu_model = poptorch.inferenceModel(Model(), opts)

    x = torch.tensor([1., 2.])
    y = torch.tensor([3., 4.])

    condition = torch.tensor([True])
    ipu_res = ipu_model(condition, x, y)
    exp_res = (x + y, (x + y) * y)
    for a, b in zip(ipu_res, exp_res):
        assert all(a == b)
