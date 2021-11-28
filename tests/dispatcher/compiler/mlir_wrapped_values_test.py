#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import pytest
import helpers
import poptorch


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_simple_adder():
    t1 = torch.randn([20])
    t2 = torch.randn([20])

    def foo(t1, t2):
        return t1 + t2

    with poptorch.IPUScope([t1, t2],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        out = foo(t1, t2)
        ipu.outputs([out])

    ipu_result = ipu(t1, t2)
    cpu_result = foo(t1, t2)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result, actual=cpu_result)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_zero_inplace():
    t1 = torch.randn([20])

    with poptorch.IPUScope([t1],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        t1.zero_()
        ipu.outputs([t1])

    ipu_result = ipu(t1)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result, actual=torch.zeros(20))


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_simple_inplace():
    t1 = torch.randn([20])

    def foo(t1):
        t1 *= 10.0

    with poptorch.IPUScope([t1],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        foo(t1)
        ipu.outputs([t1])

    ipu_result = ipu(t1)
    foo(t1)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result, actual=t1)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_simple_inplace_add():
    t1 = torch.randn([20])

    def foo(t1):
        t1 += 10.0

    with poptorch.IPUScope([t1],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        foo(t1)
        ipu.outputs([t1])

    ipu_result = ipu(t1)
    foo(t1)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result, actual=t1)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_add_with_alpha():
    t1 = torch.randn([20])

    def foo(t1):
        t1.add_(10.0, alpha=0.5)

    with poptorch.IPUScope([t1],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        foo(t1)
        ipu.outputs([t1])

    ipu_result = ipu(t1)
    foo(t1)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result, actual=t1)


@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_sub_with_alpha():
    t1 = torch.randn([20])

    def foo(t1):
        t1.sub_(10.0, alpha=0.5)

    with poptorch.IPUScope([t1],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        foo(t1)
        ipu.outputs([t1])

    ipu_result = ipu(t1)
    foo(t1)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result, actual=t1)


# TODO: Non-doubles.
@pytest.mark.skipif(not poptorch.hasMlirSupportOnPlatform(),
                    reason="CentOS 7 is not currently supported in MLIR.")
def test_wrapped_values():
    t1 = torch.randn([20])

    def foo(t1):
        t1 += 5.0
        t1 *= 2.0
        i = 6.0 / 3.0
        return t1 * i

    with poptorch.IPUScope([t1],
                           compile_using=poptorch.enums.Compiler.MLIR) as ipu:
        out = foo(t1)
        ipu.outputs([out])

    ipu_result = ipu(t1)
    cpu_result = foo(t1)

    # pylint: disable=no-member
    helpers.assert_allclose(expected=ipu_result, actual=cpu_result)
