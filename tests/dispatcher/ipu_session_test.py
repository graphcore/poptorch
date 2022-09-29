#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import copy
import pytest

import torch
from torch import nn
import poptorch
from poptorch.experimental import ipu_wrapper
import helpers


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x + 42


class SimpleModelTwo(torch.nn.Module):
    def forward(self, a, b):
        return a * b + 42


@helpers.overridePoptorchLogLevel("TRACE")
@pytest.mark.mlirSupportRequired
def test_host_buffers_no_dispatcher(capfd):
    """Test to ensure we can copy in to IPU tensors, even when the dispatcher is
    not active. When this happens, we store a copy of the tensor in a host
    buffer, which is then uploaded to the device upon execution when required
    by a model."""
    torch.manual_seed(42)

    m = torch.nn.Linear(in_features=2, out_features=4)

    # TODO(T59880): rename XLA -> IPU
    ipu_m = copy.deepcopy(m)
    ipu_m.to("xla")

    # TODO(T59880): rename XLA -> IPU
    assert ipu_m.weight.device.type == "xla"
    assert ipu_m.bias.device.type == "xla"

    # Check to see we've copied those over in to the host buffers exactly.
    helpers.assert_allequal(actual=ipu_m.weight.to("cpu"), expected=m.weight)
    helpers.assert_allequal(actual=ipu_m.bias.to("cpu"), expected=m.bias)

    lc = helpers.LogChecker(capfd)
    lc.assert_contains("Intercepting aten::copy_")
    lc.assert_contains("copy_ IPU -> CPU, outside dispatch")


@pytest.mark.mlirSupportRequired
def test_cast_no_dispatcher():
    """Test to ensure we can cast IPU tensors outside the dispatcher."""
    torch.manual_seed(42)

    x = torch.randint(100, (2, 5))

    # Float
    y = x.float()

    # TODO(T59880): rename XLA -> IPU
    ipu_x = x.int().to("xla")
    ipu_y = ipu_x.float()

    assert ipu_y.dtype == torch.float
    helpers.assert_allequal(actual=ipu_y.to("cpu"), expected=y)

    # Half
    y = x.half()

    # TODO(T59880): rename XLA -> IPU
    ipu_x = x.int().to("xla")
    ipu_y = ipu_x.half()

    assert ipu_y.dtype == torch.half
    helpers.assert_allequal(actual=ipu_y.to("cpu"), expected=y)


@helpers.overridePoptorchLogLevel("TRACE")
@pytest.mark.mlirSupportRequired
def test_weights_to_host(capfd):
    """Test we can bring individual weights off the device whilst the dispatcher
    is still active."""
    torch.manual_seed(42)

    m = torch.nn.Linear(in_features=2, out_features=4)
    ipu_m = copy.deepcopy(m)
    ipu_m.to("xla")

    x1 = torch.randn((2, ))
    y = m(x1)

    @ipu_wrapper
    def ipu_func(x):
        return ipu_m(x)

    lc = helpers.LogChecker(capfd)
    lc.assert_contains("copy_ CPU -> IPU")

    weight = ipu_m.weight.to("cpu")
    bias = ipu_m.bias.to("cpu")

    lc = helpers.LogChecker(capfd)
    lc.assert_contains("copy_ IPU -> CPU, outside dispatch")

    helpers.assert_allclose(actual=weight, expected=m.weight)
    helpers.assert_allclose(actual=bias, expected=m.bias)
    helpers.assert_allclose(actual=ipu_func(x1.to("xla")).to("cpu"),
                            expected=y)


@pytest.mark.mlirSupportRequired
def test_weights_to_device():
    param = torch.tensor(1, dtype=torch.int, device='xla')
    x = torch.tensor(3, dtype=torch.int, device='xla')

    @ipu_wrapper
    def f(x):
        nonlocal param
        param *= 2
        return x + param

    y = f(x).to('cpu')

    assert y.item() == 5
    assert param.to('cpu').item() == 2


@pytest.mark.mlirSupportRequired
def test_changing_parameters_on_host():
    pytest.skip("TODO(T69899): Parameters currently aren't reuploaded to the "
                "device if they have been changed on the host")
    param = torch.tensor(1, dtype=torch.int, device='xla')
    x = torch.tensor(3, dtype=torch.int, device='xla')

    @ipu_wrapper
    def f(x):
        nonlocal param
        param *= 2
        return x + param

    f(x).to('cpu')
    assert param.to('cpu').item() == 2

    param = torch.tensor(3.0, dtype=torch.int, device='xla')
    f(x).to('cpu')
    assert param.to('cpu').item() == 6


@pytest.mark.mlirSupportRequired
def test_changing_parameters_on_device():
    pytest.skip("TODO(T69899): RegisterAtenOverloads.cpp:211: "
                "'poptorch_cpp_error': !getHostBuffer(*impl)")

    param = torch.tensor(1, dtype=torch.int, device='xla')
    x = torch.tensor(3, dtype=torch.int, device='xla')

    @ipu_wrapper
    def f(x):
        nonlocal param
        param *= 2
        return x + param

    @ipu_wrapper
    def g(x):
        nonlocal param
        param = torch.tensor(3, dtype=torch.int, device=param.device)
        return param + x

    f(x).to('cpu')
    assert param.to('cpu').item() == 2

    g(x)
    assert param.to('cpu').item() == 3

    f(x).to('cpu')
    assert param.to('cpu').item() == 6


@helpers.overridePoptorchLogLevel("TRACE")
@pytest.mark.mlirSupportRequired
def test_weights_to_host_after_switch(capfd):
    """Test we bring the weights off the device before we switch executable."""
    torch.manual_seed(42)

    m = torch.nn.Linear(in_features=2, out_features=4)
    ipu_m1 = copy.deepcopy(m)
    ipu_m1.to("xla")

    x1 = torch.randn((2, ))
    _ = m(x1)

    @ipu_wrapper
    def ipu_func1(x):
        return ipu_m1(x)

    x1 = x1.to("xla")
    _ = ipu_func1(x1)

    lc = helpers.LogChecker(capfd)
    # As this will cause an early weights-to-host
    lc.assert_not_contains("copy_ IPU -> CPU")
    lc.assert_not_contains("Copying weights to host")

    class AnotherModel(torch.nn.Module):
        def forward(self, x):
            return x + 42

    ipu_m2 = AnotherModel().to("xla")

    @ipu_wrapper
    def ipu_func2(x):
        return ipu_m2(x)

    _ = ipu_func2(x1)

    lc = helpers.LogChecker(capfd)
    lc.assert_contains("Copying weights to host")

    weight = ipu_m1.weight.to("cpu")
    bias = ipu_m1.bias.to("cpu")

    lc = helpers.LogChecker(capfd)
    lc.assert_contains("Ignored copyWeightsToHost: not needed")

    helpers.assert_allclose(actual=weight, expected=m.weight)
    helpers.assert_allclose(actual=bias, expected=m.bias)


@pytest.mark.mlirSupportRequired
def test_copy_tensor():
    """Ensure we can copy tensors outside of the session and still use them."""

    m = SimpleModel()
    ipu_m = copy.deepcopy(m)
    ipu_m.to("xla")

    x1 = torch.randn((2, ))
    y = m(x1)
    x2 = x1.to("xla")
    x3 = copy.deepcopy(x2)

    ipu_y = ipu_wrapper(ipu_m)(x3)

    helpers.assert_allclose(actual=ipu_y.to("cpu"), expected=y)


@pytest.mark.mlirSupportRequired
def test_detach():
    """Ensure we can detach tensors outside of the IPUSession and use either
    tensor. When PyTorch does `to.device`, it also causes a detach so this is
    required to allow conversion outside of the dispatcher."""

    m = SimpleModelTwo()
    ipu_m = copy.deepcopy(m)
    ipu_m.to("xla")

    x1 = torch.randn((2, ))
    y = m(x1, x1)
    x2 = x1.to("xla")
    x3 = x2.detach()

    @ipu_wrapper
    def ipu_func(a, b):
        return ipu_m(a, b)

    helpers.assert_allclose(actual=ipu_func(x2, x3).to("cpu"), expected=y)


@pytest.mark.mlirSupportRequired
def test_args_and_kwargs():
    """Ensure both args and keyword arguments are passed to the executable
    correctly."""

    m = SimpleModelTwo()
    ipu_m = copy.deepcopy(m)
    ipu_m.to("xla")

    x = torch.randn((2, ))
    y = m(x, x)

    ipu_x = x.to("xla")

    @ipu_wrapper
    def ipu_func1(a, b, **kwargs):  # pylint: disable=unused-argument
        return ipu_m(a, b)

    helpers.assert_allclose(actual=ipu_func1(ipu_x, ipu_x, c=ipu_x).to("cpu"),
                            expected=y)

    # Call again: use the cached executable.
    helpers.assert_allclose(actual=ipu_func1(ipu_x, ipu_x, c=ipu_x).to("cpu"),
                            expected=y)

    @ipu_wrapper
    def ipu_func2(a, b, **kwargs):
        return a * b + kwargs["c"]

    y = x * x + x

    helpers.assert_allclose(actual=ipu_func2(ipu_x, ipu_x, c=ipu_x).to("cpu"),
                            expected=y)

    # Call again: use the cached executable.
    helpers.assert_allclose(actual=ipu_func2(ipu_x, ipu_x, c=ipu_x).to("cpu"),
                            expected=y)


class MNISTBlock(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, pool_size):
        super(MNISTBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              num_filters,
                              kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x


class MNISTNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = MNISTBlock(1, 10, 5, 2)
        self.layer2 = MNISTBlock(10, 20, 5, 2)
        self.layer3 = nn.Linear(320, 256, False)
        self.layer3_act = nn.ReLU()
        self.layer4 = nn.Linear(256, 10)

        self.softmax = nn.LogSoftmax(1)
        self.loss = nn.NLLLoss(reduction="mean")

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(-1, 320)

        x = self.layer3_act(self.layer3(x))

        x = self.layer4(x)
        x = self.softmax(x)
        return x


@pytest.mark.mlirSupportRequired
def test_mnist():
    model = MNISTNetwork()
    input = torch.ones([1, 1, 28, 28])

    ipu_model = copy.deepcopy(model)
    ipu_model.to("xla")

    @ipu_wrapper
    def ipu_func(x):
        return ipu_model(x)

    helpers.assert_allclose(expected=model(input),
                            actual=ipu_func(input.to("xla")).to("cpu"),
                            atol=1e-05,
                            rtol=1e-05,
                            equal_nan=True)


@pytest.mark.mlirSupportRequired
def test_compiler_options():
    """Test passing in the CompilerOptions."""

    ipu_m = SimpleModel()
    ipu_m.to("xla")

    x = torch.randn((2, )).to("xla")

    @ipu_wrapper(compiler_options=poptorch.CompilerOptions())
    def ipu_func(inp):
        return ipu_m(inp)

    # TODO(georgew): mock this once CompilerOptions is used within `ipu_wrapper`

    _ = ipu_func(x)


@pytest.mark.mlirSupportRequired
def test_no_tensor_arguments():
    """Ensure calling an ipu_wrapper-wrapped function fails when no tensor
    arguments are provided: otherwise, this will fail later on with a strange
    error claiming that the model is being recompiled when it shouldn't be."""
    ipu_model = SimpleModel()
    ipu_model.to("xla")

    @ipu_wrapper()
    def ipu_func1():
        return ipu_model()

    with pytest.raises(poptorch.Error, match="No tensor inputs passed"):
        ipu_func1()

    @ipu_wrapper()
    def ipu_func2(inp):
        return ipu_model(inp)

    with pytest.raises(poptorch.Error, match="No tensor inputs passed"):
        ipu_func2(42)


@pytest.mark.mlirSupportRequired
def test_function_reuse():
    @ipu_wrapper
    def f(x):
        return x + 1

    in1 = torch.tensor(1, dtype=torch.int, device='xla')
    out1 = f(in1)

    in2 = torch.tensor(2, dtype=torch.int, device='xla')
    out2 = f(in2)

    assert out1.to('cpu').item() == 2
    assert out2.to('cpu').item() == 3
