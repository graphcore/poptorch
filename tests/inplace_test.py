#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import re
import torch
import torch.nn as nn
import pytest
import helpers
import poptorch


def test_inplace_add():
    class Model(nn.Module):
        def forward(self, x):
            if isinstance(x, (tuple, list)):
                x[0] += 4
            elif isinstance(x, (dict)):
                x['input'] += 3
            else:
                x += 1

    poptorch_model = poptorch.inferenceModel(Model())
    tensor_in = torch.Tensor([1.0])
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 2.0
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 3.0
    assert poptorch_model(torch.Tensor([1.0])) is None
    assert tensor_in == 3.0

    # We're changing the input type: must recompile
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


def test_inplace_add_multi_elements():
    class Model(nn.Module):
        def forward(self, _x, y):
            y += 1

    poptorch_model = poptorch.inferenceModel(Model())
    nested_tuple_in = ((torch.Tensor([1.0]), torch.Tensor([1.0])),
                       (torch.Tensor([1.0])))
    tensor_in = torch.Tensor([1.0])

    assert poptorch_model(nested_tuple_in, tensor_in) is None
    assert tensor_in == 2.0


def test_inplace_sub():
    class Model(nn.Module):
        def forward(self, x):
            if isinstance(x, (tuple, list)):
                x[0] -= 3
            elif isinstance(x, (dict)):
                x['input'] -= 2
            else:
                x -= 1

    poptorch_model = poptorch.inferenceModel(Model())
    tensor_in = torch.Tensor([1.0])
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 0.0
    assert poptorch_model(tensor_in) is None
    assert tensor_in == -1.0
    assert poptorch_model(torch.Tensor([1.0])) is None
    assert tensor_in == -1.0

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


def test_inplace_div():
    class Model(nn.Module):
        def forward(self, x):
            if isinstance(x, (tuple, list)):
                x[0] /= 4
            elif isinstance(x, (dict)):
                x['input'] /= 3
            else:
                x /= 2

    poptorch_model = poptorch.inferenceModel(Model())
    tensor_in = torch.Tensor([1.0])
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 0.5
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 0.25
    assert poptorch_model(torch.Tensor([1.0])) is None
    assert tensor_in == 0.25

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


def test_inplace_mul():
    class Model(nn.Module):
        def forward(self, x):
            if isinstance(x, (tuple, list)):
                x[0] *= 4
            elif isinstance(x, (dict)):
                x['input'] *= 3
            else:
                x *= 2

    poptorch_model = poptorch.inferenceModel(Model())
    tensor_in = torch.Tensor([1.0])
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 2.0
    assert poptorch_model(tensor_in) is None
    assert tensor_in == 4.0
    assert poptorch_model(torch.Tensor([1.0])) is None
    assert tensor_in == 4.0

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


def test_inplace_masked_fill():
    class Model(nn.Module):
        def forward(self, x):
            x.masked_fill_(x > 0.5, 1.0)

    poptorch_model = poptorch.inferenceModel(Model())
    x = torch.tensor([[0, 0.7], [0.2, 3.5]])
    poptorch_model(x)

    assert x[0][0] == 0
    assert x[0][1] == 1.0
    assert x[1][0] == 0.2
    assert x[1][1] == 1.0


def test_chained_inplace():
    class Model(nn.Module):
        def forward(self, x, y):
            x += y
            x += 2.0
            x += y

    model = Model()
    t1 = torch.tensor([1.])
    cpu_t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])
    poptorch_model = poptorch.inferenceModel(model)
    out = model(cpu_t1, t2)
    assert out is None
    out = poptorch_model(t1, t2)
    assert out is None
    assert cpu_t1 == 7.0
    assert t1 == 7.0


def test_inplace_zero():
    class Model(nn.Module):
        def forward(self, x):
            # (Simply setting it to zero gets pruned by PopART)
            a = torch.sum(x)
            x.zero_()
            x += a

    poptorch_model = poptorch.inferenceModel(Model())
    x = torch.tensor([[0, 0.5], [0.25, 2.0]])
    poptorch_model(x)

    assert x[0][0] == 2.75
    assert x[0][1] == 2.75
    assert x[1][0] == 2.75
    assert x[1][1] == 2.75


def test_inplace_fill():
    class Model(nn.Module):
        def forward(self, x):
            a = torch.sum(x)
            x.fill_(1.0)
            x += a

    poptorch_model = poptorch.inferenceModel(Model())
    x = torch.tensor([[0, 0.5], [0.25, 2.0]])
    poptorch_model(x)

    assert x[0][0] == 3.75
    assert x[0][1] == 3.75
    assert x[1][0] == 3.75
    assert x[1][1] == 3.75


def test_inplace_non_input():
    class Model(nn.Module):
        def forward(self, x):
            a = x + 1
            a += 1
            return a

    poptorch_model = poptorch.inferenceModel(Model())
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


def test_double_underscore():
    # This tests aten::__and__ is not treated as inplace

    class Model(nn.Module):
        def forward(self, x, l):

            return x[0].int() & l.int()

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)
    inp, l = torch.rand(10, 10), torch.LongTensor([10])

    out = model(inp, l)
    popout = poptorch_model(inp, l)

    helpers.assert_allclose(actual=popout, expected=out)


def test_half_buffer_inplace():
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
    poptorch_model = poptorch.inferenceModel(model)

    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float16)
    out = poptorch_model(x)

    helpers.assert_allclose(actual=out,
                            expected=torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5],
                                                  dtype=torch.float16))
    poptorch_model.copyWeightsToHost()
    helpers.assert_allclose(actual=poptorch_model.buff,
                            expected=torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0],
                                                  dtype=torch.float16))


def test_float_to_half_buffer_inplace_with_training():
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
    poptorch_model = poptorch.trainingModel(model)

    x = torch.rand(5, 5).half()
    native_out, native_loss = model(x)

    # Reset buff
    model.buff = torch.ones(5, 5)

    poptorch_out, poptorch_loss = poptorch_model(x)

    helpers.assert_allclose(actual=native_out, expected=poptorch_out)
    helpers.assert_allclose(actual=native_loss, expected=poptorch_loss)


def test_inplace_on_buffer_and_input():
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

    model = poptorch.inferenceModel(Model())

    buf, out = model(torch.ones(shape))

    expected_out = torch.full(shape, fill_value)
    expected_buf = expected_out + 1

    helpers.assert_allequal(actual=out, expected=expected_out)
    helpers.assert_allequal(actual=buf, expected=expected_buf)


def test_two_inplace_copies():
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

    model = poptorch.inferenceModel(Model())

    out = model(torch.ones(shape))

    expected_out = torch.full(shape, fill_value) + 3

    helpers.assert_allequal(actual=out, expected=expected_out)


def test_two_inplace_copies_buffer():
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

    model = poptorch.inferenceModel(Model())

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
    x *= 2
    x[0:2:step.item()] = x[0:2:step.item()] * 0
    return x


def modify_region(x, step):
    x[1:x.shape[0]:step.item(), :] += 1
    return x


@pytest.mark.parametrize("step_size", [1, 2])
@pytest.mark.parametrize("op", [
    direct_assign, direct_assign_inplace, direct_fill, chained_slice,
    modify_before_assign, modify_region
])
def test_inplace_modify_slice(op, step_size):
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
        try:
            ipu_model.compile(t)
        except poptorch.Error as e:
            assert re.match(
                r"In\-place modification of slices with step "
                r"size other than 1 is not supported\.", e.message)


def test_inplace_modify_select():
    shape = (3, 4, 2)

    inpA = torch.randint(55, shape)
    inpB = torch.randint(66, shape)
    inpC = torch.randint(77, shape)

    class ModelWrapper(torch.nn.Module):
        def forward(self, tensorA, tensorB, tensorC):
            tensorA = tensorA - tensorB

            tensorA[0:1] += tensorC[1]
            tensorA[0] += tensorC[0]
            tensorA[1][2] += tensorC[2][1]
            tensorA[1][3][1] += tensorC[2][3][0]

            return tensorA

    model = ModelWrapper()

    cpu_out = model(inpA, inpB, inpC)

    poptorch_model = poptorch.inferenceModel(model)
    ipu_out = poptorch_model(inpA, inpB, inpC)

    helpers.assert_allclose(actual=ipu_out, expected=cpu_out)


def test_index_put_on_buffer():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            p_init = torch.arange(6, dtype=torch.float).reshape(2, 3)
            self.register_buffer("p", p_init)

        def forward(self, x, idx):
            self.p[(idx, )] = x
            return self.p

    model = Model()
    ipu_model = poptorch.inferenceModel(Model())

    x = torch.empty(3).fill_(-1)
    idx = torch.tensor([0])
    cpu_out = model(x, idx)
    ipu_out = ipu_model(x, idx)
    helpers.assert_allclose(actual=ipu_out, expected=cpu_out)
