#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import re
import torch
import torch.nn as nn
import pytest
import poptorch
import helpers


@pytest.mark.parametrize("use_half", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_simple_tuple(use_half, trace_model):
    class SimpleAdder(nn.Module):
        def forward(self, t):
            assert isinstance(t, tuple)
            (x, y) = t
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            return x + y

    model = SimpleAdder()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])

    if use_half:
        model.half()
        t1 = t1.half()
        t2 = t2.half()
    assert inference_model((t1, t2)).float() == 3.0
    # Run more than once
    assert inference_model((t1, t2)).float() == 3.0


@pytest.mark.parametrize("trace_model", [True, False])
def test_type_change(trace_model):
    class SimpleAdder(nn.Module):
        def forward(self, t):
            assert isinstance(t, tuple)
            (x, y) = t
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            return x + y

    model = SimpleAdder()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    assert inference_model((t1, t2)).float() == 3.0
    # Run more than once
    assert inference_model((t1, t2)).float() == 3.0

    t1 = torch.tensor([1])
    t2 = torch.tensor([2])
    error_msg = (".*expected torch.float32 but got torch.int64.*")
    with pytest.raises(poptorch.Error, match=error_msg):
        assert inference_model((t1, t2)).float() == 3

    inference_model.destroy()
    assert inference_model((t1, t2)).float() == 3


@pytest.mark.parametrize("trace_model", [True, False])
def test_shape_change(trace_model):
    class SimpleAdder(nn.Module):
        def forward(self, t):
            assert isinstance(t, tuple)
            (x, y) = t
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            return x + y

    model = SimpleAdder()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    assert inference_model((t1, t2)).float() == 3.0
    # Run more than once
    assert inference_model((t1, t2)).float() == 3.0

    t1 = torch.tensor([1., 1.])
    t2 = torch.tensor([2., 2.])
    error_msg = ("expected torch.Size([1]) but got torch.Size([2])")
    with pytest.raises(poptorch.Error, match=re.escape(error_msg)):
        assert inference_model((t1, t2)).float() == 3

    inference_model.destroy()
    native_out = model((t1, t2))
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = inference_model((t1, t2))
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("use_half", [True, False])
@pytest.mark.parametrize("thing_to_test", ['List', 'Tuple', 'Mixed'])
@pytest.mark.parametrize("trace_model", [True, False])
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_nested_tuples_and_lists(capfd, use_half, thing_to_test, trace_model):
    if trace_model and thing_to_test in ["List", "Mixed"]:
        pytest.skip("Tracing doesn't support lists, only tuples")

    class SimpleAdder(nn.Module):
        def forward(self, tpl1, t2, tpl34567):
            (t1, ) = tpl1
            (t3, (t4, t5), _) = tpl34567
            (t6, _) = tpl34567[2]
            t7 = tpl34567[2][1]

            assert isinstance(t1, torch.Tensor)
            assert isinstance(t2, torch.Tensor)
            assert isinstance(t3, torch.Tensor)
            assert isinstance(t4, torch.Tensor)
            assert isinstance(t5, torch.Tensor)
            assert isinstance(t6, torch.Tensor)
            assert isinstance(t7, torch.Tensor)

            return t1 + t2 + t3 + t4 + t5 + t6 + t7

    model = SimpleAdder()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    t3 = torch.tensor([3.])
    t4 = torch.tensor([4.], dtype=torch.float64)
    t5 = torch.tensor([5.])
    t6 = torch.tensor([6.])
    t7 = torch.tensor([7.], dtype=torch.float64)

    if use_half:
        model.half()
        t1 = t1.half()
        t2 = t2.half()
        t3 = t3.half()
        t4 = t4.half()
        t5 = t5.half()
        t6 = t6.half()
        t7 = t7.half()

    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        if thing_to_test == "List":
            assert inference_model([
                t1,
            ], t2, [t3, [t4, t5], [t6, t7]]).float() == 28.0
        elif thing_to_test == "Tuple":
            assert inference_model((t1, ), t2,
                                   (t3, (t4, t5), (t6, t7))).float() == 28.0
        else:
            assert inference_model([
                t1,
            ], t2, [t3, (t4, t5), [t6, t7]]).float() == 28.0

    # Ensure that a tuple element's type is not changed except after the
    # "host_side_cast"
    if trace_model and not use_half:
        testlog = helpers.LogChecker(capfd)
        testlog.assert_contains_after(
            "Double(1, strides=[1], requires_grad=0, device=cpu) = " +
            "prim::TupleUnpack", "Graph right before popart")


@pytest.mark.parametrize("use_half", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_optional_inputs(use_half, trace_model):
    dtype = torch.float16 if use_half else torch.float32

    class SimpleAdder(nn.Module):
        def forward(self,
                    t1,
                    t2,
                    t3=torch.ones(1, dtype=dtype),
                    t4=torch.zeros(1, dtype=dtype)):
            return t1 * t3 + t2 * t4

    model = SimpleAdder()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    t4 = torch.tensor([4.])

    if use_half:
        model.half()
        t1 = t1.half()
        t2 = t2.half()
        t4 = t4.half()

    assert inference_model(t1, t2).float() == 1.0
    assert inference_model(t1, t2, t4=t4).float() == 9.0
    assert inference_model(t4=t4, t1=t1, t2=t2).float() == 9.0


@pytest.mark.mlirSupportRequired
def test_non_tensor_inputs_dispatch():
    class Model(nn.Module):
        def forward(
                self,
                t1,
                scalar=2,
                t2_opt=None,
        ):
            if t2_opt is not None:
                return t2_opt * scalar + t1 * scalar
            return t1 * scalar

    model = Model()
    options = poptorch.Options()
    # jit.trace() cannot support this.
    options.Jit.traceModel(False)

    t1 = torch.tensor([3.])
    ipu = poptorch.inferenceModel(model, options)(t1)
    cpu = model(t1)
    helpers.assert_allclose(expected=cpu, actual=ipu)

    scalar = 4
    ipu = poptorch.inferenceModel(model, options)(t1, scalar)
    cpu = model(t1, scalar)
    helpers.assert_allclose(expected=cpu, actual=ipu)

    t2 = torch.tensor([5.])
    ipu = poptorch.inferenceModel(model, options)(t1, scalar, t2)
    cpu = model(t1, scalar, t2)
    helpers.assert_allclose(expected=cpu, actual=ipu)

    ipu = poptorch.inferenceModel(model, options)(t1, t2_opt=t2)
    cpu = model(**{"t1": t1, "t2_opt": t2})
    helpers.assert_allclose(expected=cpu, actual=ipu)


@pytest.mark.parametrize("use_half", [True, False])
@pytest.mark.parametrize("trace_model", [True, False])
def test_list_inputs(use_half, trace_model):
    class SimpleAdder(nn.Module):
        def forward(self, t1, t2, x):
            l = [t1, t2]
            x = l[0] + x
            l[1] = x
            return l

    model = SimpleAdder()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    t3 = torch.tensor([4.])

    if use_half:
        model.half()
        t1 = t1.half()
        t2 = t2.half()
        t3 = t3.half()

    expected = [torch.tensor([1.0]), torch.tensor([5.0])]

    assert [t.float() for t in inference_model(t1, t2, t3)] == expected

    # Call multiple times to check the fast path works
    assert [t.float() for t in inference_model(t1, t2, t3)] == expected
    assert [t.float() for t in inference_model(t1, t2, t3)] == expected
    assert [t.float() for t in inference_model(t1, t2, t3)] == expected


@pytest.mark.parametrize("trace_model", [True, False])
def test_unused_tuple(trace_model):
    class SimpleAdder(nn.Module):
        def forward(self, x, y, z):  # pylint: disable=unused-argument
            return x + y

    model = SimpleAdder()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)
    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    z = (torch.tensor([1.]), torch.tensor([1.]))
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        inference_model(t1, t2, z)


@pytest.mark.parametrize("trace_model", [True, False])
def test_input_plain_dict(trace_model):
    model = torch.nn.ReLU()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    with pytest.raises(TypeError) as excinfo:
        inference_model({'a': torch.tensor([1])})

    assert (str(
        excinfo.value) == "Dictionaries are not supported as input arguments, "
            "including when nested in tuples.\n"
            "Received dict input = {'a': tensor([1])}")


@pytest.mark.parametrize("trace_model", [True, False])
def test_input_nested_dict(trace_model):
    model = torch.nn.ReLU()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    with pytest.raises(TypeError) as excinfo:
        inference_model(
            (torch.tensor([1]), torch.tensor([2]), (torch.tensor([3]), {
                'b': torch.tensor([4])
            }, torch.tensor([5])), torch.tensor([6])))

    assert (str(excinfo.value) == "Dictionaries are not supported as input "
            "arguments, including when nested in tuples."
            "\nReceived dict input[2][1] = "
            "{'b': tensor([4])}")


@pytest.mark.parametrize("trace_model", [True, False])
def test_input_three_agg_nested_dict(trace_model):
    class ThreeIdentity(nn.Module):
        def forward(self, x, y, z):
            return x, y, z

    model = ThreeIdentity()

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)

    with pytest.raises(TypeError) as excinfo:
        inference_model(torch.tensor([0]),
                        (torch.tensor([1]), torch.tensor([2]),
                         (torch.tensor([3]), {
                             'c': torch.tensor([4])
                         }, torch.tensor([5])), torch.tensor([6])),
                        torch.tensor([7]))

    assert (str(excinfo.value) == "Dictionaries are not supported as input "
            "arguments, including when nested in tuples."
            "\nReceived dict y[2][1] = "
            "{'c': tensor([4])}")


torch.manual_seed(42)
ones = torch.ones(5, 5)
x = torch.randn(5, 5)
y = torch.randn(5, 5)
z = torch.randn(5, 5)
t = torch.randn(5, 5)


class Model(torch.nn.Module):
    def forward(self, x, y=None, z=None, t=None):
        r = x
        if y is not None:
            r = torch.add(r, y) * 3
        if z is not None:
            r = torch.add(r, z) * 4
        if t is not None:
            r = torch.add(r, t) * 5
        return torch.tanh(r)


@pytest.mark.parametrize("trace_model", [True, False])
def test_none_input_pass_one_kwarg(trace_model):
    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)

    native_out = model(x, y, z, t=None)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(x, y, z, t=None)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("trace_model", [True, False])
def test_none_input_pass_two_kwarg(trace_model):
    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)

    native_out = model(x, y, z=None, t=None)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(x, y, z=None, t=None)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("trace_model", [True, False])
def test_none_input_pass_skip_one_kwarg(trace_model):
    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)

    native_out = model(x, y, z=None)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(x, y, z=None)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


def test_none_input_trace_fail_non_default_kwarg():
    class Model(torch.nn.Module):
        def forward(self, x, y=torch.ones(2, 2)):
            return x + y

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(True)
    poptorch_model = poptorch.inferenceModel(model, options)

    with pytest.raises(poptorch.Error, match="'None' may not be passed"):
        poptorch_model(x, y=None)


@pytest.mark.mlirSupportRequired
def test_none_input_trace_dispatch_non_default_kwarg():
    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(False)
    poptorch_model = poptorch.inferenceModel(model, options)

    native_out = model(x, y=None)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(x, y=None)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("trace_model", [True, False])
def test_none_input_pass_last_arg(trace_model):
    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)

    native_out = model(x, y, z, None)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(x, y, z, None)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("trace_model", [True, False])
def test_none_input_pass_two_arg(trace_model):
    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)

    native_out = model(x, y, None, None)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(x, y, None, None)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("args", [(x, None, None, None), (x, ), (x, None)])
@pytest.mark.parametrize("fwd_args", [True, False])
def test_none_input_dispatch_non_default_arg_tuples(args, fwd_args):
    class ModelWrapper(Model):
        def forward(self, *args, **kwargs):  # pylint: disable=signature-differs
            return super().forward(*args, **kwargs)

    if fwd_args:
        model = ModelWrapper()
    else:
        model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(False)
    poptorch_model = poptorch.inferenceModel(model, options)

    native_out = model(*args)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(*args)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("args", [{
    "x": x,
    "t": t
}, {
    "z": z,
    "t": None,
    "x": x
}])
@pytest.mark.parametrize("fwd_args", [True, False])
def test_none_input_dispatch_non_default_arg_dict(args, fwd_args):
    class ModelWrapper(Model):
        def forward(self, *args, **kwargs):  # pylint: disable=signature-differs
            return super().forward(*args, **kwargs)

    if fwd_args:
        model = ModelWrapper()
    else:
        model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(False)
    poptorch_model = poptorch.inferenceModel(model, options)

    native_out = model(**args)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(**args)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.mlirSupportRequired
@pytest.mark.parametrize("fwd_args", [True, False])
def test_none_input_dispatch_args_kwargs(fwd_args):
    class Model(torch.nn.Module):
        def forward(self, a, b, *c, y=None, z=None, t=None, u=3, v="op", **w):
            r = len(v) * b + a * len(w)
            for i, x in enumerate(c):
                r += (i + 1) * x
            if y is not None:
                r = torch.add(r, y) * 3
            if z is not None:
                r = torch.add(r, z) * 4
            if t is not None:
                r = torch.add(r, t) * 5
            return u * r

    class ModelWrapper(Model):
        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs)

    if fwd_args:
        model = ModelWrapper()
    else:
        model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(False)
    poptorch_model = poptorch.inferenceModel(model, options)

    a = 2
    b = torch.randn(2, 2)
    c = torch.randn(2, 2)
    d = torch.randn(2, 2)
    e = torch.randn(2, 2)
    t = torch.randn(2, 2)
    x = torch.randn(2, 2)
    m = torch.randn(2, 2)
    z = torch.randn(2, 2)

    native_out = model(a, b, c, d, e, t=t, x=x, m=m, z=z)
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(a, b, c, d, e, t=t, x=x, m=m, z=z)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)

    if fwd_args:
        expected = "Missing arguments: z."
    else:
        expected = "Type mismatch for z: expected .*Tensor.* but got .*None"
    with pytest.raises(poptorch.Error, match=expected):
        poptorch_out = poptorch_model(a, b, c, d, e, t=t, x=x, m=m)

    with pytest.raises(poptorch.Error, match="Missing arguments: m."):
        poptorch_out = poptorch_model(a, b, c, d, e, t=t, x=x, z=z)

    poptorch_model.destroy()
    native_out = model(a, b, c, d, e, t=t, x=x, m=m, z=z, u=5, v="foobar")
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(a,
                                      b,
                                      c,
                                      d,
                                      e,
                                      t=t,
                                      x=x,
                                      m=m,
                                      z=z,
                                      u=5,
                                      v="foobar")
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)

    with pytest.raises(poptorch.Error,
                       match="mismatch for u: expected 5 but got 3"):
        poptorch_out = poptorch_model(a,
                                      b,
                                      c,
                                      d,
                                      e,
                                      t=t,
                                      x=x,
                                      m=m,
                                      z=z,
                                      u=3,
                                      v="foobar")

    with pytest.raises(
            poptorch.Error,
            match=("Number of positional arguments mismatch: expected"
                   " 5 arguments but got 4")):
        poptorch_model(a, b, c, e, t=t, x=x, m=m, z=z, u=5, v="foobar")

    with pytest.raises(
            poptorch.Error,
            match=("Number of positional arguments mismatch: expected "
                   "5 arguments but got 2")):
        poptorch_model(a, b, t=t, x=x, m=m, z=z, u=5, v="foobar")

    poptorch_model.destroy()
    if fwd_args:
        error_type = TypeError
        error = "missing 1 required positional argument: 'b'"
    else:
        error_type = poptorch.Error
        error = "Mandatory parameter b missing"

    with pytest.raises(error_type, match=error):
        poptorch_model(a)

    native_out = model(a, b)
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(a, b)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


def test_none_input_trace_fail_non_default_arg():
    class Model(torch.nn.Module):
        def forward(self, x, y=torch.ones(2, 2)):
            return x + y

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(True)
    poptorch_model = poptorch.inferenceModel(model, options)

    with pytest.raises(poptorch.Error, match="'None' may not be passed"):
        poptorch_model(x, None)


def test_none_input_fail():
    class Model(torch.nn.Module):
        def forward(self, x=None, y=ones):
            if x is None:
                return y
            if y is None:
                return ones
            return torch.add(x, y)

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(True)
    poptorch_model = poptorch.inferenceModel(model, options)

    with pytest.raises(poptorch.Error, match="'None' may not be passed"):
        poptorch_model(x, None)


def test_no_inputs():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.tensor([1.], dtype=torch.float)

        def forward(self):
            self.x += 1.0
            return self.x

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(True)
    poptorch_model = poptorch.inferenceModel(model, options)

    # It appears that forward is called enough time as to make the value 7 as
    # part of the tracing.

    assert poptorch_model() == 7.
    assert poptorch_model() == 7.


@pytest.mark.parametrize("trace_model", [True, False])
def test_no_inputs_no_output(trace_model):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.tensor([1.], dtype=torch.float)

        def forward(self):
            self.x += self.x

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    poptorch_model()
    poptorch_model()


@pytest.mark.parametrize("trace_model", [True, False])
def test_return_and_use_input(trace_model):
    class Model(torch.nn.Module):
        def forward(self, input):
            c = torch.tensor([1.])
            return c, input + c

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    assert poptorch_model(torch.tensor([0.])) == (torch.tensor([1.]),
                                                  torch.tensor([1.]))
    assert poptorch_model(torch.tensor([1.])) == (torch.tensor([1.]),
                                                  torch.tensor([2.]))


@pytest.mark.parametrize("trace_model", [True, False])
def test_return_and_use_nested_input(trace_model):
    class Model(torch.nn.Module):
        def forward(self, input):
            c = torch.tensor([1.])

            c = poptorch.set_available_memory(c, 0.1)

            return c, (c, input + c)

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    assert poptorch_model(torch.tensor([0.])) == (torch.tensor([1.]),
                                                  (torch.tensor([1.]),
                                                   torch.tensor([1.])))
    assert poptorch_model(torch.tensor([1.])) == (torch.tensor([1.]),
                                                  (torch.tensor([1.]),
                                                   torch.tensor([2.])))
