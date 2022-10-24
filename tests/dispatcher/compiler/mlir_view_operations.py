#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest
import torch
import helpers
from poptorch.experimental import IPUContext


def harness(op, **kwargs):
    torch.manual_seed(42)
    t = torch.randn((5, 2, 1))

    ipu_result = IPUContext(op)(t, **kwargs)
    cpu_result = op(t, **kwargs)

    helpers.assert_allclose(expected=cpu_result, actual=ipu_result)


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
def test_view(capfd):
    def f(x):
        y = torch.reshape(x, (2, 5))
        x += 1
        y += 1
        return x

    harness(f)

    checker = helpers.LogChecker(capfd)
    checker.assert_contains('poptorch.viewOutplace')


def test_nested_views():
    def f(x):
        y = torch.reshape(x, (2, 5))
        z = torch.reshape(x, (5, 2))
        z += 1
        y += 1
        x += 1
        return x

    harness(f)


def test_view_with_outplace():
    def f(x):
        y = torch.reshape(x, (2, 5))
        z = y + 1
        y += 2
        return x, z

    harness(f)


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
def test_squeeze_(capfd):
    def f(x):
        x.squeeze_()
        x += 1
        return x

    harness(f)

    checker = helpers.LogChecker(capfd)
    checker.assert_contains('poptorch.viewOutplace')


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
def test_nested_squeeze_(capfd):
    def g(x):
        y = x.reshape((5, 2, 1))
        z = y.squeeze_()
        z += 1
        return x

    harness(g)

    checker = helpers.LogChecker(capfd)
    checker.assert_contains('poptorch.viewOutplace')


@helpers.overridePoptorchLogLevel("DEBUG")
def test_inplace_copy_with_squeeze():
    def h(x):
        t = x.clone()
        t += 1
        z = t.squeeze()

        y = x.squeeze()
        y.copy_(z)

        return x

    harness(h)


def expand(x, *args):
    return x.expand(*args)


def slice_pos(x):
    return x[1:4:2]


def as_strided(x, *args):
    return x.as_strided(*args)


view_ops = [(torch.reshape, ((2, 5), )), (torch.transpose, (0, 1)),
            (torch.permute, ((1, 0, 2), )), (expand, ((2, 5, 2, 1), )),
            (torch.squeeze, ()), (torch.squeeze, (2, )),
            (torch.unsqueeze, (1, )), (torch.select, (2, 0)), (slice_pos, ()),
            (as_strided, ((5, 2), (2, 1), 0)), (torch.detach, ()),
            (torch.detach_, ())]


@pytest.mark.parametrize("view_op, args", view_ops)
@pytest.mark.parametrize("inplace", [False, True])
def test_all_view_ops(view_op, args, inplace):
    if inplace and view_op in [expand]:
        pytest.skip(
            f'Changing view inplace not currently implemented for {view_op}')

    def f(x):
        y = view_op(x, *args)
        x += 1
        if inplace:
            y += 1
        else:
            y = y + 1
        return x, y

    harness(f)


def test_chained_view_ops():
    text_len = 81
    b = 3

    def f(mask):
        mask = mask[:, :text_len]
        mask = mask.view([3, 81])
        mask = mask.permute([0, 1])
        mask = mask.unsqueeze(1)
        mask = mask.unsqueeze(2)
        mask = mask.unsqueeze(3)
        mask = mask.expand([-1, 16, 16, 16, -1])
        mask = mask.reshape([48, 16, 16, 81])
        return mask

    ipu_res = IPUContext(f)(torch.ones(b, text_len).to(torch.bool))
    cpu_res = f(torch.ones(b, text_len).to(torch.bool))

    helpers.assert_allequal(actual=ipu_res, expected=cpu_res)


def test_chained_slice_and_index():
    def f(x, y):
        x[:, :, 0] = y
        return x

    x = torch.ones(2, 3, 5)
    y = torch.zeros(2, 3)

    ipu_res = IPUContext(f)(x, y)
    cpu_res = f(x, y)

    helpers.assert_allequal(actual=ipu_res, expected=cpu_res)


def test_chained_slice():
    def f(x):
        t = x.reshape(3, 3, 3)
        s1 = t[:, :, :1]  # shape = [3, 3, 2]
        s1 += 1
        s2 = s1[:, :1]  # shape = [3, 2, 2]
        s2 += 2
        s3 = s2[:1]  # shape = [2, 2, 2]
        s3 += 3
        return s1, s2, s3

    x = torch.arange(27)

    ipu_res = IPUContext(f)(x)
    cpu_res = f(x)

    helpers.assert_allequal(actual=ipu_res, expected=cpu_res)
