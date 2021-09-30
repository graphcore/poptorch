#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import poptorch
import helpers


def test_print_tensor():
    class Model(torch.nn.Module):
        def forward(self, x):
            return poptorch.ipu_print_tensor(x)

    m = poptorch.inferenceModel(Model())
    m(torch.randn(5))


def test_print_tensor_with_title():
    class Model(torch.nn.Module):
        def forward(self, x):
            return poptorch.ipu_print_tensor(x, "my_tensor")

    m = poptorch.inferenceModel(Model())
    m(torch.randn(5))


def test_nop():
    class Model(torch.nn.Module):
        def forward(self, x):
            return poptorch.nop(x) * 2

    m = poptorch.inferenceModel(Model())
    m(torch.randn(5))


def test_name_scope():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            with poptorch.NameScope("NameScope"):
                return x + y

    model = Model()
    poptorch_model = poptorch.inferenceModel(model)

    torch.manual_seed(42)
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)
    poptorch_model(x, y)

    ir = poptorch_model._debugGetPopartIR()  # pylint: disable=protected-access
    assert ir.find('"name":"NameScope/Add:InPlace"') != -1


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_available_memory_last_op(capfd):
    class Model(torch.nn.Module):
        def forward(self, x):
            x = torch.matmul(x, x)
            return poptorch.set_available_memory(x, 0.3)

    input = torch.randn(10, 10)
    poptorch_model = poptorch.inferenceModel(Model())
    poptorch_model.compile(input)

    # Check the trace log to make sure set_available_memory isn't pruned
    # before it's lowered to PopART
    ir_before_popart_regex = \
    (r"Graph right before popart:\n"
     r".*\n"
     r".* popart::matmul.*\n"
     r".* poptorch::set_available_memory.*")

    log = helpers.LogChecker(capfd)
    log.assert_matches(ir_before_popart_regex, per_line=False)
