#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import collections
import re
import torch
import torch.nn as nn
import pytest
import helpers
import poptorch


@pytest.mark.parametrize("use_half", [True, False])
def test_simple_tuple(use_half):
    class SimpleAdder(nn.Module):
        def forward(self, t):
            assert isinstance(t, tuple)
            (x, y) = t
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            return x + y

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])

    if use_half:
        model.half()
        t1 = t1.half()
        t2 = t2.half()
    assert inference_model((t1, t2)).float() == 3.0
    # Run more than once
    assert inference_model((t1, t2)).float() == 3.0


def test_type_change():
    class SimpleAdder(nn.Module):
        def forward(self, t):
            assert isinstance(t, tuple)
            (x, y) = t
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            return x + y

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

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


def test_shape_change():
    class SimpleAdder(nn.Module):
        def forward(self, t):
            assert isinstance(t, tuple)
            (x, y) = t
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            return x + y

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

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
def test_nested_tuples_and_lists(use_half, thing_to_test):
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
    inference_model = poptorch.inferenceModel(model)

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


@pytest.mark.parametrize("use_half", [True, False])
def test_optional_inputs(use_half):
    dtype = torch.float16 if use_half else torch.float32

    class SimpleAdder(nn.Module):
        def forward(self,
                    t1,
                    t2,
                    t3=torch.ones(1, dtype=dtype),
                    t4=torch.zeros(1, dtype=dtype)):
            return t1 * t3 + t2 * t4

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

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

    t1 = torch.tensor([3.])
    ipu = poptorch.inferenceModel(model)(t1)
    cpu = model(t1)
    helpers.assert_allclose(expected=cpu, actual=ipu)

    scalar = 4
    ipu = poptorch.inferenceModel(model)(t1, scalar)
    cpu = model(t1, scalar)
    helpers.assert_allclose(expected=cpu, actual=ipu)

    t2 = torch.tensor([5.])
    ipu = poptorch.inferenceModel(model)(t1, scalar, t2)
    cpu = model(t1, scalar, t2)
    helpers.assert_allclose(expected=cpu, actual=ipu)

    ipu = poptorch.inferenceModel(model)(t1, t2_opt=t2)
    cpu = model(**{"t1": t1, "t2_opt": t2})
    helpers.assert_allclose(expected=cpu, actual=ipu)


@pytest.mark.parametrize("use_half", [True, False])
def test_list_inputs(use_half):
    class SimpleAdder(nn.Module):
        def forward(self, t1, t2, x):
            l = [t1, t2]
            x = l[0] + x
            l[1] = x
            return l

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)

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


def test_unused_tuple():
    class SimpleAdder(nn.Module):
        def forward(self, x, y, z):  # pylint: disable=unused-argument
            return x + y

    model = SimpleAdder()
    inference_model = poptorch.inferenceModel(model)
    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    z = (torch.tensor([1.]), torch.tensor([1.]))
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        inference_model(t1, t2, z)


def test_dict_input():
    class DictDivider(nn.Module):
        def forward(self, d):  # pylint: disable=unused-argument
            return d['x'] / d['y']

    model = DictDivider()
    z = {'x': torch.tensor([1.]), 'y': torch.tensor([2.])}
    native_out = model(z)
    inference_model = poptorch.inferenceModel(model)

    # Run more than once
    for i in range(4):
        # Reorder the dict to check order doesn't matter
        if i == 1:
            z = {'y': torch.tensor([2.]), 'x': torch.tensor([1.])}
        # Missing argument
        elif i == 2:
            z = {'y': torch.tensor([2.])}
            with pytest.raises(poptorch.Error, match="Missing arguments: x."):
                inference_model(z)
            continue
        # Extra argument
        elif i == 3:
            z = {
                'x': torch.tensor([1.]),
                'y': torch.tensor([2.]),
                'z': torch.tensor([3.])
            }
            with pytest.raises(poptorch.Error,
                               match="Unexpected arguments: z."):
                inference_model(z)
            continue
        poptorch_out = inference_model(z)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


def test_nested_dict_input():
    class DictAdder(nn.Module):
        def forward(self, d):  # pylint: disable=unused-argument
            return d[0]['d']['x'] + d[0]['d']['y'] + d[1]

    model = DictAdder()
    z = [{
        'd': {
            'x': torch.tensor([1.]),
            'y': torch.tensor([2.])
        }
    },
         torch.tensor([3.])]
    native_out = model(z)
    inference_model = poptorch.inferenceModel(model)

    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = inference_model(z)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


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


def test_none_input_pass_one_kwarg():
    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    native_out = model(x, y, z, t=None)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(x, y, z, t=None)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


def test_none_input_pass_two_kwarg():
    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    native_out = model(x, y, z=None, t=None)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(x, y, z=None, t=None)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


def test_none_input_pass_skip_one_kwarg():
    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    native_out = model(x, y, z=None)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(x, y, z=None)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


def test_none_input_trace_dispatch_non_default_kwarg():
    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    native_out = model(x, y=None)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(x, y=None)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


def test_none_input_pass_last_arg():
    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    native_out = model(x, y, z, None)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(x, y, z, None)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


def test_none_input_pass_two_arg():
    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    native_out = model(x, y, None, None)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(x, y, None, None)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


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
    poptorch_model = poptorch.inferenceModel(model)

    native_out = model(*args)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(*args)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


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
    poptorch_model = poptorch.inferenceModel(model)

    native_out = model(**args)
    # Run more than once
    for i in range(2):
        print(f"Run {i}")
        poptorch_out = poptorch_model(**args)
        helpers.assert_allclose(expected=native_out, actual=poptorch_out)


@pytest.mark.parametrize("fwd_args", [True, False])
def test_custom_arg_parser(fwd_args):
    class MyArg:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    class MyParser(poptorch.ICustomArgParser):
        def yieldTensors(self, struct) -> None:
            yield struct.a
            yield struct.b

        def reconstruct(self, _original_structure, tensor_iterator):
            return MyArg(next(tensor_iterator), next(tensor_iterator))

    class OutputContainer(collections.OrderedDict):
        def print(self):
            return str(self)

    poptorch.registerCustomArgParser(MyArg, MyParser())

    class Model(torch.nn.Module):
        def forward(self, args):
            # Make sure to use a poptorch specific op
            # to check the graph is not empty or running on the CPU
            out = OutputContainer()
            out["sum"] = args.a + poptorch.ipu_print_tensor(args.b)
            out["a"] = args.a
            return out

    class ModelWrapper(Model):
        def forward(self, *args, **kwargs):
            print(len(args))
            return super().forward(*args, **kwargs)

    if fwd_args:
        model = ModelWrapper()
    else:
        model = Model()

    poptorch_model = poptorch.inferenceModel(model)

    args = MyArg(torch.randn(2, 2), torch.randn(2, 2))
    for i in range(2):
        print(f"Run {i}")
        args = MyArg(torch.randn(2, 2), torch.randn(2, 2))
        native_out = model(args)
        poptorch_out = poptorch_model(args)
        # Make sure we get an OutputContainer and the elements are in the same order
        assert isinstance(native_out, OutputContainer)
        assert isinstance(poptorch_out, OutputContainer)
        print(native_out.print())
        print(poptorch_out.print())
        for native_key, poptorch_key in zip(native_out, poptorch_out):
            assert native_key == poptorch_key
            helpers.assert_allclose(expected=native_out[native_key],
                                    actual=poptorch_out[poptorch_key])


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
            print(len(args))
            return super().forward(*args, **kwargs)

    if fwd_args:
        model = ModelWrapper()
    else:
        model = Model()
    poptorch_model = poptorch.inferenceModel(model)

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


def test_no_inputs_no_output():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.tensor([1.], dtype=torch.float)

        def forward(self):
            self.x += self.x

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_model()
    poptorch_model()


def test_return_and_use_input():
    class Model(torch.nn.Module):
        def forward(self, input):
            c = torch.tensor([1.])
            return c, input + c

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)
    assert poptorch_model(torch.tensor([0.])) == (torch.tensor([1.]),
                                                  torch.tensor([1.]))
    assert poptorch_model(torch.tensor([1.])) == (torch.tensor([1.]),
                                                  torch.tensor([2.]))


def test_return_and_use_nested_input():
    class Model(torch.nn.Module):
        def forward(self, input):
            c = torch.tensor([1.])

            c = poptorch.set_available_memory(c, 0.1)

            return c, (c, input + c)

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)
    assert poptorch_model(torch.tensor([0.])) == (torch.tensor([1.]),
                                                  (torch.tensor([1.]),
                                                   torch.tensor([1.])))
    assert poptorch_model(torch.tensor([1.])) == (torch.tensor([1.]),
                                                  (torch.tensor([1.]),
                                                   torch.tensor([2.])))


def test_scalar_tensor_input():
    class Square(torch.nn.Module):
        def forward(self, x):
            return x * x

    model = Square()
    s = poptorch.inferenceModel(model)
    x = torch.tensor(3.)  # shape = torch.Size([])
    helpers.assert_allclose(actual=s(x), expected=model(x))


def test_returned_only_inputs():
    class Model(torch.nn.Module):
        def forward(self, x, y, z):
            # x and y will be erased as inputs and converted to
            # host-side-only constants
            return x, y, z + 0.0

    m = Model()
    p = poptorch.inferenceModel(m)
    x = torch.tensor([1, 2])
    y = torch.tensor([3, 4])
    z = torch.tensor([1.2, 3.4])

    for cpu_out, ipu_out in zip(m(x, y, z), p(x, y, z)):
        helpers.assert_allclose(actual=ipu_out, expected=cpu_out)
