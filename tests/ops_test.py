#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import re
import torch
import helpers
import poptorch


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
    (r"Graph before lowering to PopART:\n"
     r".*\n"
     r".* popart::matmul.*\n"
     r".* poptorch::set_available_memory.*")

    log = helpers.LogChecker(capfd)
    log.assert_matches(ir_before_popart_regex, per_line=False)


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_available_memory_linear(capfd):
    class LinModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.lin = torch.nn.Linear(3, 3)

        def forward(self, x):
            x = self.conv(x)
            x = self.lin(x)
            x = poptorch.set_available_memory(x, 0.3)
            return x

    x = torch.rand(2, 3, 5, 5)
    model = LinModel()
    poptorch_model = poptorch.inferenceModel(model)
    poptorch_model(x)

    log = helpers.LogChecker(capfd)
    it = log.createIterator()
    # Assert that the set_available_memory node references the matmul, not the
    # add.
    it.findNext("Graph before lowering to PopART:")
    matmul_line = it.findNext("popart::matmul").strip()
    matmul_var = matmul_line.partition(" ")[0]
    sam_line = it.findNext("poptorch::set_available_memory").strip()
    actual_var = re.match(r".*set_available_memory[^\(]+\(([^\)]+).*",
                          sam_line).group(1)
    assert actual_var == matmul_var
