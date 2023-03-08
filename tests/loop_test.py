#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import pytest
import helpers
import poptorch


def test_loop_constant():
    class Model(torch.nn.Module):
        def forward(self, x):
            def body(x):
                return x * 2

            return poptorch.for_loop(10, body, [x])[0]

    inference_model = poptorch.inferenceModel(Model())

    x = torch.tensor([1.])

    assert inference_model(x) == pow(2, 10)


def test_loop_simple():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            def body(x):
                return x * y

            return poptorch.for_loop(10, body, [x])[0]

    inference_model = poptorch.inferenceModel(Model())

    x = torch.tensor([1.])
    y = torch.tensor([2.])
    assert inference_model(x, y) == pow(2, 10)


def test_loop_multiple_inputs():
    class Model(torch.nn.Module):
        def forward(self, x, y, z, w):
            def body(x, y, z, w):
                return x * y, y + z, x * w, w + 1

            return poptorch.for_loop(10, body, [x, y, z, w])

    inference_model = poptorch.inferenceModel(Model())

    x = torch.tensor([0.1])
    y = torch.tensor([0.2])
    z = torch.tensor([0.3])
    w = torch.tensor([0.4])

    out = inference_model(x, y, z, w)

    # Check by running equiv on host.
    x = torch.tensor([0.1])
    y = torch.tensor([0.2])
    z = torch.tensor([0.3])
    w = torch.tensor([0.4])

    for _ in range(0, 10):
        _z = x * w
        x *= y
        y += z
        w = w + 1
        z = _z

    for host, ipu in zip([x, y, z, w], out):
        assert host == ipu


def test_loop_non_tensor_in():
    class Model(torch.nn.Module):
        def forward(self, x, _):
            def body(x, y):
                return x * y, y + 1

            return poptorch.for_loop(10, body, [x, 5])

    inference_model = poptorch.inferenceModel(Model())

    x = torch.tensor([1.])
    y = torch.tensor([2.])

    msg = "(Object contained in list at index 1 is not torch.tensor)"
    with pytest.raises(ValueError, match=msg):
        inference_model(x, y)


def test_loop_non_list_in():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            def body(x):
                return x * y

            return poptorch.for_loop(10, body, x)

    inference_model = poptorch.inferenceModel(Model())

    x = torch.tensor([1.])
    y = torch.tensor([2.])

    msg = "(Object is not list)"
    with pytest.raises(ValueError, match=msg):
        inference_model(x, y)


def test_loop_weights():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.layer1 = torch.nn.Linear(1, 256)
            self.layer2 = torch.nn.Conv2d(4, 1, [8, 8])

        def forward(self, x):
            def body(x):
                act = self.layer1(x)
                act = act.reshape([1, 4, 8, 8])
                act = self.layer2(act)
                return act.flatten()

            return poptorch.for_loop(2, body, [x])[0]

    inference_model = poptorch.inferenceModel(Model())

    x = torch.tensor([1.])

    inference_model(x)


def test_loop_weights_use_twice():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(4, 4)

        def forward(self, x):
            def body(x):
                act = self.layer1(x)
                return self.layer1(act)

            return poptorch.for_loop(2, body, [x])

    inference_model = poptorch.inferenceModel(Model())

    x = torch.ones(1, 4).to(torch.float)
    inference_model(x)


def test_loop_use_output():
    class Model(torch.nn.Module):
        def forward(self, x):
            def body(x):
                return x + x

            out = poptorch.for_loop(2, body, [x])[0]
            loss = poptorch.identity_loss(out, reduction='sum')
            return out, loss

    inference_model = poptorch.inferenceModel(Model())

    x = torch.ones(1, 4).to(torch.float)
    inference_model(x)


def test_loop_training():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(4, 4)

        def forward(self, x):
            def body(x):
                return self.layer1(x)

            out = poptorch.for_loop(2, body, [x])[0]
            loss = poptorch.identity_loss(out, reduction='sum')
            return out, loss

    training_model = poptorch.trainingModel(Model())

    x = torch.ones(1, 4).to(torch.float)
    with pytest.raises(
            poptorch.Error,
            match=r"poptorch.for_loop\(\) is only supported in inference"):
        training_model(x)


def test_loop_body_inplace_ops_1():
    class Model(torch.nn.Module):
        def forward(self, x):
            # Body inputs are passed by value so 'x' remains unchanged.
            def body(y):
                y += 1
                return y

            return poptorch.for_loop(3, body, [x])[0]

    poptorch_model = poptorch.inferenceModel(Model())
    x = torch.ones(1, 5).to(torch.int32)
    x_copy = torch.ones(1, 5).to(torch.int32)

    out = poptorch_model(x)
    helpers.assert_allequal(actual=x, expected=x_copy)
    helpers.assert_allequal(actual=out, expected=x_copy * 4)


def test_loop_body_inplace_ops_2():
    class Model(torch.nn.Module):
        def forward(self, x):
            # Body inputs are passed by value so 'x' remains unchanged.
            def body(y):
                y += 1
                y += 1
                return y

            return poptorch.for_loop(3, body, [x])[0]

    poptorch_model = poptorch.inferenceModel(Model())
    x = torch.ones(1, 5).to(torch.int32)
    x_copy = torch.ones(1, 5).to(torch.int32)

    out = poptorch_model(x)
    helpers.assert_allequal(actual=x, expected=x_copy)
    helpers.assert_allequal(actual=out, expected=x_copy * 7)


def test_loop_body_inplace_ops_3():
    class Model(torch.nn.Module):
        def forward(self, x):
            x += 1

            # Body inputs are passed by value so 'x' remains unchanged.
            def body(y):
                y += 1
                return y

            return poptorch.for_loop(3, body, [x])[0]

    poptorch_model = poptorch.inferenceModel(Model())
    x = torch.ones(1, 5).to(torch.int32)
    x_copy = torch.ones(1, 5).to(torch.int32)

    out = poptorch_model(x)
    helpers.assert_allequal(actual=x, expected=x_copy * 2)
    helpers.assert_allequal(actual=out, expected=x_copy * 5)


def test_loop_body_inplace_ops_4():
    class Model(torch.nn.Module):
        def forward(self, x):
            x += 1

            # Body inputs are passed by value so 'x' remains unchanged.
            def body(y):
                y += 1
                return y

            z = poptorch.for_loop(3, body, [x])[0]
            x += 1
            return z

    poptorch_model = poptorch.inferenceModel(Model())
    x = torch.ones(1, 5).to(torch.int32)
    x_copy = torch.ones(1, 5).to(torch.int32)

    out = poptorch_model(x)
    helpers.assert_allequal(actual=x, expected=x_copy * 3)
    helpers.assert_allequal(actual=out, expected=x_copy * 5)


def test_loop_with_constant_inputs_only():
    class Model(torch.nn.Module):
        def forward(self):
            # 't0' will be evaluated as part of constexpr folding.
            t0 = torch.tensor([0., 0.])
            t0 = t0 + 8
            # 't1' and 't2' must not be evaluated as part of constexpr folding.
            t1 = torch.tensor([1., 2.])
            t2 = torch.tensor([3., 4.])

            def func(x, y):
                x = x * 2
                y = y * x
                return x, y

            t1, t2 = poptorch.for_loop(5, func, [t1, t2])
            return t1, t0

    poptorch_model = poptorch.inferenceModel(Model())
    helpers.assert_allequal(actual=poptorch_model(),
                            expected=(torch.tensor([32., 64.]),
                                      torch.tensor([8., 8.])))


def test_loop_with_same_trip_count_on_multiple_ipus():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            def func(x, y):
                x = x + y
                return x, y

            # Note: both trip_count equal to 5
            with poptorch.Block("0", ipu_id=0):
                x, y = poptorch.for_loop(5, func, [x, y])

            with poptorch.Block("1", ipu_id=1):
                x, y = poptorch.for_loop(5, func, [x, y])

            return x, y

    native = Model()
    stages = [poptorch.Stage(f"{k}") for k in range(0, 2)]
    strategy = poptorch.ShardedExecution(*stages)

    opts = poptorch.Options()
    opts.setExecutionStrategy(strategy)
    ipu = poptorch.inferenceModel(native, opts)

    x = torch.tensor([1., 2.])
    y = torch.tensor([1., 2.])

    ipu_out = ipu(x, y)[0]
    native_out = x + 10 * y

    helpers.assert_allclose(actual=ipu_out, expected=native_out)
