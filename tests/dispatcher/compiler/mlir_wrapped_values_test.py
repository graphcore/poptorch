#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
import poptorch
from poptorch.experimental import IPUContext


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_simple_adder():
    t1 = torch.randn([20])
    t2 = torch.randn([20])

    def add(t1, t2):
        return t1 + t2

    add_ipu = IPUContext(add)

    cpu_result = add(t1, t2)
    ipu_result = add_ipu(t1, t2)

    helpers.assert_allclose(expected=cpu_result, actual=ipu_result)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_zero_inplace():
    t = torch.randn([20])

    @IPUContext
    def ipu_zero_(t):
        t.zero_()
        return t

    helpers.assert_allclose(expected=ipu_zero_(t), actual=torch.zeros(20))


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_simple_inplace():
    t1 = torch.randn([20])

    def mul_inplace(t):
        t *= 10.0
        return t

    ipu_result = IPUContext(mul_inplace)(t1)
    mul_inplace(t1)

    helpers.assert_allclose(expected=t1, actual=ipu_result)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_simple_inplace_add():
    t1 = torch.randn([20])

    def add_inplace(t):
        t += 10.0
        return t

    ipu_result = IPUContext(add_inplace)(t1)
    add_inplace(t1)

    helpers.assert_allclose(expected=t1, actual=ipu_result)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_add_with_alpha():
    t1 = torch.randn([20])

    def add_with_alpha(t):
        t.add_(10.0, alpha=0.5)
        return t

    ipu_result = IPUContext(add_with_alpha)(t1)
    add_with_alpha(t1)

    print(ipu_result)
    print(t1)

    helpers.assert_allclose(expected=t1, actual=ipu_result)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_sub_with_alpha():
    t1 = torch.randn([20])

    def sub_with_alpha(t):
        t.sub_(10.0, alpha=0.5)
        return t

    ipu_result = IPUContext(sub_with_alpha)(t1)
    sub_with_alpha(t1)

    helpers.assert_allclose(expected=t1, actual=ipu_result)


# TODO(T49190): More than just float and long
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
@pytest.mark.parametrize("python_type", [float, int])
def test_wrapped_values(python_type):
    dtype = torch.float if python_type is float else torch.int
    t1 = torch.ones([20], dtype=dtype)

    def f(t):
        t += python_type(5)
        t *= python_type(2)
        # Use floor division so that both types work
        i = python_type(6) // python_type(3)
        return t * i

    ipu_result = IPUContext(f)(t1)
    cpu_result = f(t1)

    helpers.assert_allclose(expected=cpu_result, actual=ipu_result)
