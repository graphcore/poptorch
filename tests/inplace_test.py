#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn

import pytest
import poptorch
import helpers


@pytest.mark.parametrize("trace_model", [True, False])
def test_inplace_add(trace_model):
    class Model(nn.Module):
        def forward(self, x):
            if isinstance(x, (tuple, list)):
                x[0] += 4
            elif isinstance(x, (dict)):
                x['input'] += 3
            else:
                x += 1

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(Model(), options)
    tensor_in = torch.Tensor([1.0])
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 2.0
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 3.0
    assert poptorch_model(torch.Tensor([1.0])) is None
    assert tensor_in == 3.0

    if trace_model:
        # Tracing doesn't support lists as inputs, and tuples
        # can't be modified in-place.
        return

    # We're changing the  input type: must recompile
    poptorch_model.destroy()
    list_in = [torch.Tensor([1.0])]
    cpu_in = [torch.Tensor([1.0])]
    model = Model()
    for i in range(2):
        print(f"Run {i}")
        cpu_out = model(cpu_in)
        poptorch_out = poptorch_model(list_in)
        assert cpu_out == poptorch_out
        assert list_in == cpu_in


@pytest.mark.parametrize("trace_model", [True, False])
def test_inplace_add_multi_elements(trace_model):
    class Model(nn.Module):
        def forward(self, _x, y):
            y += 1

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(Model(), options)
    nested_tuple_in = ((torch.Tensor([1.0]), torch.Tensor([1.0])),
                       (torch.Tensor([1.0])))
    tensor_in = torch.Tensor([1.0])

    assert poptorch_model(nested_tuple_in, tensor_in) is None
    assert tensor_in == 2.0


@pytest.mark.parametrize("trace_model", [True, False])
def test_inplace_sub(trace_model):
    class Model(nn.Module):
        def forward(self, x):
            if isinstance(x, (tuple, list)):
                x[0] -= 3
            elif isinstance(x, (dict)):
                x['input'] -= 2
            else:
                x -= 1

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(Model(), options)
    tensor_in = torch.Tensor([1.0])
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 0.0
    assert poptorch_model(tensor_in) is None
    assert tensor_in == -1.0
    assert poptorch_model(torch.Tensor([1.0])) is None
    assert tensor_in == -1.0

    if trace_model:
        # Tracing doesn't support lists as inputs, and tuples
        # can't be modified in-place.
        return

    # We're changing the  input type: must recompile
    poptorch_model.destroy()
    list_in = [torch.Tensor([1.0])]
    cpu_in = [torch.Tensor([1.0])]
    model = Model()
    for i in range(2):
        print(f"Run {i}")
        cpu_out = model(cpu_in)
        poptorch_out = poptorch_model(list_in)
        assert cpu_out == poptorch_out
        assert list_in == cpu_in


@pytest.mark.parametrize("trace_model", [True, False])
def test_inplace_div(trace_model):
    class Model(nn.Module):
        def forward(self, x):
            if isinstance(x, (tuple, list)):
                x[0] /= 4
            elif isinstance(x, (dict)):
                x['input'] /= 3
            else:
                x /= 2

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(Model(), options)
    tensor_in = torch.Tensor([1.0])
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 0.5
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 0.25
    assert poptorch_model(torch.Tensor([1.0])) is None
    assert tensor_in == 0.25

    if trace_model:
        # Tracing doesn't support lists as inputs, and tuples
        # can't be modified in-place.
        return

    # We're changing the  input type: must recompile
    poptorch_model.destroy()
    list_in = [torch.Tensor([1.0])]
    cpu_in = [torch.Tensor([1.0])]
    model = Model()
    for i in range(2):
        print(f"Run {i}")
        cpu_out = model(cpu_in)
        poptorch_out = poptorch_model(list_in)
        assert cpu_out == poptorch_out
        assert list_in == cpu_in


@pytest.mark.parametrize("trace_model", [True, False])
def test_inplace_mul(trace_model):
    class Model(nn.Module):
        def forward(self, x):
            if isinstance(x, (tuple, list)):
                x[0] *= 4
            elif isinstance(x, (dict)):
                x['input'] *= 3
            else:
                x *= 2

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(Model(), options)
    tensor_in = torch.Tensor([1.0])
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 2.0
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 4.0
    assert poptorch_model(torch.Tensor([1.0])) is None
    assert tensor_in == 4.0

    if trace_model:
        # Tracing doesn't support lists as inputs, and tuples
        # can't be modified in-place.
        return

    # We're changing the  input type: must recompile
    poptorch_model.destroy()
    list_in = [torch.Tensor([1.0])]
    cpu_in = [torch.Tensor([1.0])]
    model = Model()
    for i in range(2):
        print(f"Run {i}")
        cpu_out = model(cpu_in)
        poptorch_out = poptorch_model(list_in)
        assert cpu_out == poptorch_out
        assert list_in == cpu_in


@pytest.mark.parametrize("trace_model", [True, False])
def test_inplace_masked_fill(trace_model):
    class Model(nn.Module):
        def forward(self, x):
            x.masked_fill_(x > 0.5, 1.0)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(Model(), options)
    x = torch.tensor([[0, 0.7], [0.2, 3.5]])
    poptorch_model(x)

    assert x[0][0] == 0
    assert x[0][1] == 1.0
    assert x[1][0] == 0.2
    assert x[1][1] == 1.0


@pytest.mark.parametrize("trace_model", [True, False])
def test_chained_inplace(trace_model):
    class Model(nn.Module):
        def forward(self, x, y):
            x += y
            x += 2.0
            x += y

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    model = Model()
    t1 = torch.tensor([1.])
    cpu_t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    poptorch_model = poptorch.inferenceModel(model, options)
    out = model(cpu_t1, t2)
    assert out is None
    out = poptorch_model(t1, t2)
    assert out is None
    assert cpu_t1 == 7.0
    assert t1 == 7.0


@pytest.mark.parametrize("trace_model", [True, False])
def test_inplace_zero(trace_model):
    class Model(nn.Module):
        def forward(self, x):
            # (Simply setting it to zero gets pruned by PopART)
            a = torch.sum(x)
            x.zero_()
            x += a

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(Model(), options)
    x = torch.tensor([[0, 0.5], [0.25, 2.0]])
    poptorch_model(x)

    assert x[0][0] == 2.75
    assert x[0][1] == 2.75
    assert x[1][0] == 2.75
    assert x[1][1] == 2.75


@pytest.mark.parametrize("trace_model", [True, False])
def test_inplace_fill(trace_model):
    class Model(nn.Module):
        def forward(self, x):
            a = torch.sum(x)
            x.fill_(1.0)
            x += a

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(Model(), options)
    x = torch.tensor([[0, 0.5], [0.25, 2.0]])
    poptorch_model(x)

    assert x[0][0] == 3.75
    assert x[0][1] == 3.75
    assert x[1][0] == 3.75
    assert x[1][1] == 3.75


@pytest.mark.parametrize("trace_model", [True, False])
def test_inplace_non_input(trace_model):
    class Model(nn.Module):
        def forward(self, x):
            a = x + 1
            a += 1
            return a

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(Model(), options)
    x = torch.tensor([[0, 0.5], [0.25, 2.0]])

    y = poptorch_model(x)

    assert x[0][0] == 0
    assert x[0][1] == 0.5
    assert x[1][0] == 0.25
    assert x[1][1] == 2.0

    assert y[0][0] == 2
    assert y[0][1] == 2.5
    assert y[1][0] == 2.25
    assert y[1][1] == 4.0


@pytest.mark.parametrize("trace_model", [True, False])
def test_double_underscore(trace_model):
    # This tests aten::__and__ is not treated as inplace

    class Model(nn.Module):
        def forward(self, x, l):

            return x[0].int() & l.int()

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)
    inp, l = torch.rand(10, 10), torch.LongTensor([10])

    out = model(inp, l)
    popout = poptorch_model(inp, l)

    helpers.assert_allclose(actual=popout, expected=out)


@pytest.mark.parametrize("trace_model", [True, False])
def test_half_buffer_inplace(trace_model):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('buff', torch.ones(5, dtype=torch.float16))

        def forward(self, x):
            # pylint: disable=no-member
            out = x + self.buff
            self.buff += 1
            return out

    model = Model()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(model, options)

    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float16)
    out = poptorch_model(x)

    helpers.assert_allclose(actual=out,
                            expected=torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5],
                                                  dtype=torch.float16))
    poptorch_model.copyWeightsToHost()
    helpers.assert_allclose(actual=poptorch_model.buff,
                            expected=torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0],
                                                  dtype=torch.float16))


@pytest.mark.parametrize("trace_model", [True, False])
def test_float_to_half_buffer_inplace_with_training(trace_model):
    torch.manual_seed(42)

    # pylint: disable=attribute-defined-outside-init
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

            # need at least one parameter for a training model
            self.param = nn.Parameter(torch.ones(5, 5))

            self.register_buffer("buff", torch.ones(5))
            self.loss = nn.MSELoss()

        def forward(self, x):
            # pylint: disable=no-member
            out = self.buff + self.param
            self.buff += 1
            return out, self.loss(out, x)

    model = Model().train().half()
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.trainingModel(model, options)

    x = torch.rand(5, 5).half()
    native_out, native_loss = model(x)

    # Reset buff
    model.buff = torch.ones(5, 5)

    poptorch_out, poptorch_loss = poptorch_model(x)

    helpers.assert_allclose(actual=native_out, expected=poptorch_out)
    helpers.assert_allclose(actual=native_loss, expected=poptorch_loss)


@pytest.mark.parametrize("trace_model", [True, False])
def test_inplace_on_buffer_and_input(trace_model):
    fill_value = 3
    shape = (1, 2)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.ones(shape))

        def forward(self, x):
            # Perform inplace ops on both the input and our buffer.
            x.fill_(fill_value)

            buffer_update = self.buffer + x
            self.buffer.copy_(buffer_update)

            return self.buffer, x

    opts = poptorch.Options()
    opts.Jit.traceModel(trace_model)

    model = poptorch.inferenceModel(Model(), opts)

    buf, out = model(torch.ones(shape))

    expected_out = torch.full(shape, fill_value)
    expected_buf = expected_out + 1

    helpers.assert_allequal(actual=out, expected=expected_out)
    helpers.assert_allequal(actual=buf, expected=expected_buf)


@pytest.mark.parametrize("trace_model", [True, False])
def test_two_inplace_copies(trace_model):
    fill_value = 3
    shape = (1, 2)

    class Model(torch.nn.Module):
        def forward(self, x):
            res = torch.full(shape, fill_value)
            x.copy_(res)

            # Do a second `copy_` to our input.
            res += 3
            x.copy_(res)

            return x

    opts = poptorch.Options()
    opts.Jit.traceModel(trace_model)

    model = poptorch.inferenceModel(Model(), opts)

    out = model(torch.ones(shape))

    expected_out = torch.full(shape, fill_value) + 3

    helpers.assert_allequal(actual=out, expected=expected_out)


@pytest.mark.parametrize("trace_model", [True, False])
def test_two_inplace_copies_buffer(trace_model):
    fill_value = 3
    shape = (1, 2)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.ones(shape))

        def forward(self, x):
            x.fill_(fill_value)

            buffer_update = self.buffer + x
            self.buffer.copy_(buffer_update)

            # Do a second `copy_` to our buffer.
            buffer_update += 5
            self.buffer.copy_(buffer_update)

            return self.buffer, x

    opts = poptorch.Options()
    opts.Jit.traceModel(trace_model)

    model = poptorch.inferenceModel(Model(), opts)

    buf, out = model(torch.ones(shape))

    expected_out = torch.full(shape, fill_value)
    expected_buf = expected_out + 6

    helpers.assert_allequal(actual=out, expected=expected_out)
    helpers.assert_allequal(actual=buf, expected=expected_buf)


def direct_assign(x, step):
    x[0:2:step.item()] = x[0:2:step.item()] * 0
    return x


def direct_assign_inplace(x, step):
    x[0:2:step.item()] *= 0
    return x


def direct_fill(x, step):
    x[0:2:step.item()] = 0
    return x


# Slicing entire dimensions lowers to slice(slice(x))
def chained_slice(x, step):
    x[:, :2:step.item()].mul_(0)
    return x


def modify_before_assign(x, step):
    x = x * 2
    x[0:2:step.item()] = x[0:2:step.item()] * 0
    return x


@helpers.printCapfdOnExit
@pytest.mark.parametrize("step_size", [1, 2])
@pytest.mark.parametrize("op", [
    direct_assign, direct_assign_inplace, direct_fill, chained_slice,
    modify_before_assign
])
def test_inplace_modify_slice(op, step_size, capfd):
    t = torch.rand(4, 4)
    step = torch.tensor(step_size)

    class Model(torch.nn.Module):
        pass

    Model.forward = lambda _, x: op(x, step)

    cpu_model = Model()
    ipu_model = poptorch.inferenceModel(cpu_model)

    if step_size == 1:
        ipu_input = t.clone()
        cpu_input = t.clone()
        # Ensure outputs match
        helpers.assert_allclose(actual=ipu_model(ipu_input),
                                expected=cpu_model(cpu_input))
        # Ensure that any inplace modification of graph inputs is
        # correctly reflected
        helpers.assert_allclose(actual=ipu_input, expected=cpu_input)
    else:
        if op is direct_fill:
            with pytest.raises(
                    poptorch.poptorch_core.Error,
                    match="All operations in the main graph were pruned, "
                    "nothing to compute"):
                ipu_model.compile(t)
        else:
            ipu_model.compile(t)

        testlog = helpers.LogChecker(capfd)
        testlog.assert_matches(
            r"\[warning\] In\-place modification of slices with step "
            r"size other than 1 is not supported\. This may result in "
            r"unexpected behaviour\.")
