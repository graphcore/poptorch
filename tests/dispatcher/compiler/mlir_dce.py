#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest
import torch
import helpers
from poptorch.experimental import IPUContext


@pytest.mark.mlirSupportRequired
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
def test_dead_code_elimination(capfd):
    def func(x):
        t = x * x  # pylint: disable=unused-variable
        return x

    ipu_out = IPUContext(func)(torch.tensor(2))

    helpers.assert_allequal(expected=torch.tensor(2), actual=ipu_out)

    checker = helpers.LogChecker(capfd)
    checker.assert_not_contains('poptorch.mul')


@pytest.mark.mlirSupportRequired
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("DEBUG")
def test_dead_code_elimination_with_views(capfd):
    def func(x):
        y = x * 2
        s = x.reshape((2, 2))
        s += s
        return y

    input = torch.tensor([1, 2, 3, 4])
    ipu_out = IPUContext(func)(input)

    helpers.assert_allequal(expected=input * 2, actual=ipu_out)

    checker = helpers.LogChecker(capfd)
    checker.assert_not_contains('poptorch.add')
