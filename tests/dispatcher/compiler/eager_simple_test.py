#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import inspect
import torch
import torchvision.models as models
import pytest
import helpers


def simple_add(capfd):
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    capfd.readouterr()  # Clear the log

    def fn(input, check_log=False, first_compile=False):
        x = input + 5
        if check_log:
            # Check the add was lowered and executed.
            log = helpers.LogChecker(capfd)
            log.assert_not_contains("CPU -> IPU")
            log.assert_not_contains("IPU -> CPU")
            if first_compile:
                log.assert_contains("Graph lowered to popit")
            else:
                log.assert_contains("reusing PopIT function")

        return x * 3

    input = torch.ones([10])
    cpu = fn(input)
    log = helpers.LogChecker(capfd)
    log.assert_isEmpty()
    input = input.to("xla")
    log = helpers.LogChecker(capfd)
    log.assert_contains("CPU -> IPU")
    ipu = fn(input, check_log=True, first_compile=True)
    # Check the multiplication was also lowered and executed.
    log = helpers.LogChecker(capfd)
    log.assert_not_contains("CPU -> IPU")
    log.assert_not_contains("IPU -> CPU")
    log.assert_contains("Graph lowered to popit")
    ipu = fn(input, check_log=True)
    # Check that we cached the PopIT function created on the last run and reused
    # it on this run.
    log = helpers.LogChecker(capfd)
    log.assert_not_contains("CPU -> IPU")
    log.assert_not_contains("IPU -> CPU")
    log.assert_contains("Found graph in cache")
    print(f"Result cpu: {cpu} ipu: {ipu}")
    # Check the print triggered a copy to host
    log = helpers.LogChecker(capfd)
    log.assert_not_contains("CPU -> IPU")
    log.assert_contains("IPU -> CPU")
    helpers.assert_allclose(expected=cpu, actual=ipu.cpu())
    # Check .cpu() triggered a copy to host
    log = helpers.LogChecker(capfd)
    log.assert_not_contains("CPU -> IPU")
    log.assert_contains("IPU -> CPU")


@pytest.mark.ipuHardwareRequired
@helpers.overridePoptorchLogLevel("DEBUG")
@pytest.mark.parametrize("mode", ["default", "show_all", "hide_all"])
def test_source_location(mode):
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    layer = torch.nn.Linear(1, 2).to('xla')
    expected_filename = inspect.stack()[0].filename
    # +3 -> We expect to see f()'s return line in the log
    expected_line = inspect.stack()[0].lineno + 3

    def f(x):
        return layer(x)

    if mode == "show_all":
        # Clear the list: show everything
        poptorch.eager.eager_options.source_location_excludes = []
    elif mode == "hide_all":
        # All paths have a '/' in them so we essentially exclude everything.
        poptorch.eager.eager_options.source_location_excludes += ['/']

    input = torch.Tensor([[1.], [-1.]]).to('xla')
    res = f(input)

    log = helpers.LogChecker(poptorch.poptorch_core.getCachedGraph(res))

    # By default: we point at the user code
    default_loc = r'loc\("' + f'{expected_filename}":{expected_line}'
    # If we clear the list of exclusions we will point at Torch's internals
    torch_internal_loc = "site-packages/torch/nn/functional.py"
    unknown_loc = r"\#loc = loc\(unknown\)"
    if mode == "show_all":
        log.assert_matches(torch_internal_loc)
        log.assert_no_matches(default_loc)
    elif mode == "hide_all":
        log.assert_matches(unknown_loc)
        log.assert_no_matches(torch_internal_loc)
        log.assert_no_matches(default_loc)
    else:
        log.assert_no_matches(torch_internal_loc)
        log.assert_matches(default_loc)


@pytest.mark.ipuHardwareRequired
@helpers.overridePoptorchLogLevel("TRACE")
def test_unchanged_output_removal():
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    t = torch.arange(6).to('xla')
    s = t.reshape(2, 3).clone()

    log = helpers.LogChecker(poptorch.poptorch_core.getCachedGraph(s))
    log.assert_no_matches(r'return .*\%arg\d')


@pytest.mark.ipuHardwareRequired
@helpers.overridePoptorchLogLevel("TRACE")
def test_unused_input_removal():
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    poptorch.eager.eager_options.use_lazy_tensor = True

    t = torch.arange(6).to('xla')
    v = torch.tensor(1).to('xla')

    t.reshape(2, 3)
    u = v + v
    poptorch.eager.markStep()

    log = helpers.LogChecker(poptorch.poptorch_core.getCachedGraph(u))
    log.assert_no_matches(r'func.func @MainGraph.*tensor<6xsi32>')


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("INFO")
@pytest.mark.extendedTestingOnly
def test_lazy_tensor(capfd):
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    poptorch.eager.eager_options.use_lazy_tensor = True

    t = torch.tensor(1.0).to('xla')
    s = t + t

    log = helpers.LogChecker(capfd)
    log.assert_no_matches("Executed PopIT function")

    s.cpu()

    log = helpers.LogChecker(capfd)
    log.assert_matches("Executed PopIT function")

    s = t + t
    poptorch.eager.markStep()

    log = helpers.LogChecker(capfd)
    log.assert_matches("Executed PopIT function")


@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_simple_add(capfd):
    pytest.skip("PopIT doesn't currently support IPUModel")
    simple_add(capfd)


@pytest.mark.ipuHardwareRequired
@helpers.printCapfdOnExit
@helpers.overridePoptorchLogLevel("TRACE")
def test_simple_add_hw(capfd):
    simple_add(capfd)


@pytest.mark.ipuHardwareRequired
@pytest.mark.extendedTestingOnly
def test_casting():
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    t = torch.tensor([1], dtype=torch.int32, device='xla')
    s = t.float().to('cpu')

    assert s.dtype is torch.float
    assert s.item() == 1.0


@pytest.mark.ipuHardwareRequired
def test_view_output():
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    t = torch.arange(6)
    s = t.to('xla').reshape(2, 3).to('cpu')

    helpers.assert_allequal(expected=t.reshape(2, 3), actual=s)


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.extendedTestingOnly
def test_backward(lazy):
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel
    poptorch.eager.eager_options.use_lazy_tensor = lazy

    t = torch.tensor([1.0], requires_grad=True)
    t_x = t.to('xla')
    s = 2.0 * t_x

    s.backward()
    assert t.grad == 2.0


@pytest.mark.ipuHardwareRequired
@pytest.mark.extendedTestingOnly
def test_squeezenet():
    pytest.skip("TODO(T67125): Allocation error: out of memory")

    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    input = torch.randn([1, 3, 224, 224])

    model = models.squeezenet1_1(pretrained=False)
    model.eval()

    cpu = model(input)

    model.to("xla")
    input = input.to("xla")

    ipu = model(input)

    print(f"Result cpu: {cpu} ipu: {ipu}")
    helpers.assert_allclose(expected=cpu, actual=ipu.cpu())


@pytest.mark.ipuHardwareRequired
@pytest.mark.extendedTestingOnly
def test_resnet18():
    pytest.skip("TODO(T67125): Allocation error: out of memory")

    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel

    input = torch.randn([1, 3, 224, 224])

    model = models.resnet18(pretrained=False)
    model.eval()

    cpu = model(input)

    model.to("xla")
    input = input.to("xla")

    ipu = model(input)

    print(f"Result cpu: {cpu} ipu: {ipu}")
    helpers.assert_allclose(expected=cpu, actual=ipu.cpu())


@pytest.mark.ipuHardwareRequired
@helpers.overridePoptorchLogLevel("TRACE")
def test_no_unused_empty_tensor(capfd):
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel
    poptorch.eager.eager_options.use_lazy_tensor = True

    torch.manual_seed(42)
    x = torch.randn((10, ))

    def f(x):
        return x**2 + 5

    cpu_y = f(x)
    ipu_y = f(x.to("xla")).to("cpu")

    log = helpers.LogChecker(capfd)
    log.assert_not_contains("empty_tensor")

    helpers.assert_allclose(expected=cpu_y, actual=ipu_y)
