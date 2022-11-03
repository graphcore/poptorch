# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Callable, List, Optional
import torch
import pytest
import helpers


class Checkpoint:
    __slots__ = ["graph_hash", "log"]

    def __init__(self, graph_hash: str, log: helpers.LogChecker):
        self.graph_hash = graph_hash
        self.log = log

    def assert_new_function(self):
        self.log.assert_contains("compiling new PopIT function")

    def assert_reused_function(self):
        self.log.assert_contains("reusing PopIT function")


def harness(fn: Callable[[torch.Tensor, Callable[[], None]], torch.Tensor],
            x: torch.Tensor,
            capfd: Optional[pytest.CaptureFixture],
            atol: float = 1e-5,
            rtol: float = 1e-4) -> List[Checkpoint]:
    """
    Test harness for eager mode function caching. Pass None as `capfd` to
    prevent log capture - useful for `printCapfdOnExit`.
    """

    def checkpoint_cpu():
        pass

    cpu_y = fn(x, checkpoint_cpu)
    log = None
    if capfd:
        log = helpers.LogChecker(capfd)
        log.assert_isEmpty()

    checkpoints: List[Checkpoint] = []

    def checkpoint():
        nonlocal capfd, checkpoints
        if capfd:
            log = helpers.LogChecker(capfd)
            matches = log.findall(r"Graph hash is ([0-9]+)")
            assert len(matches) > 0, "Expected a graph hash, but there was " \
                "no graph hash found in the log"
            assert len(
                matches) == 1, "Bad checkpoint: more than one graph found"
            graph_hash = matches[0]
            checkpoints.append(Checkpoint(graph_hash, log))

    ipu_y = fn(x.to("xla"), checkpoint)

    helpers.assert_allclose(actual=ipu_y.to("cpu"),
                            expected=cpu_y,
                            check_dtype=True,
                            atol=atol,
                            rtol=rtol)

    return checkpoints


@helpers.overridePoptorchLogLevel("TRACE")
@pytest.mark.ipuHardwareRequired
def test_cache_simple(capfd):
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel
    torch.manual_seed(42)

    capfd.readouterr()  # Clear the log

    def fn(input, checkpoint):
        for _ in range(3):
            x = input + 5
            checkpoint()
        x = input * 3
        checkpoint()
        return x

    cp = harness(fn, torch.randn((10, )), capfd)

    assert cp[0].graph_hash == cp[0].graph_hash
    assert cp[0].graph_hash == cp[2].graph_hash
    assert cp[0].graph_hash != cp[3].graph_hash
    cp[0].assert_new_function()
    cp[1].assert_reused_function()
    cp[2].assert_reused_function()
    cp[3].assert_new_function()


@helpers.overridePoptorchLogLevel("TRACE")
@pytest.mark.ipuHardwareRequired
def test_different_dtypes_and_shapes(capfd):
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel
    torch.manual_seed(42)

    capfd.readouterr()  # Clear the log

    def fn(input, checkpoint):
        x = input + 1
        checkpoint()
        x = input[4:] + 1
        checkpoint()
        input = input.float()
        checkpoint()
        x = input + 1
        checkpoint()
        x = input + 1
        checkpoint()
        x = input + 1
        checkpoint()
        return x

    cp = harness(fn, torch.randint(100, size=(10, ), dtype=torch.int), capfd)

    for i in range(4):
        if i > 0:
            assert cp[i - 1].graph_hash != cp[i].graph_hash
        cp[i].assert_new_function()

    assert cp[3].graph_hash == cp[4].graph_hash
    assert cp[3].graph_hash == cp[5].graph_hash
    cp[4].assert_reused_function()
    cp[5].assert_reused_function()


@helpers.overridePoptorchLogLevel("TRACE")
@pytest.mark.ipuHardwareRequired
def test_lazy_tensor(capfd):
    import poptorch.eager  # pylint: disable=unused-import, import-outside-toplevel
    poptorch.eager.eager_options.use_lazy_tensor = True

    torch.manual_seed(42)
    x = torch.randn((10, ))

    def f(x):
        return x**2 + 5

    f(x.to("xla")).to("cpu")
    log = helpers.LogChecker(capfd)
    graph_hash_1, = log.findall(r"Graph hash is ([0-9]+)")
    log.assert_contains("compiling new PopIT function")

    # Different shapes will cause recompilation
    f(x[4:].to("xla")).to("cpu")
    log = helpers.LogChecker(capfd)
    graph_hash_2, = log.findall(r"Graph hash is ([0-9]+)")
    log.assert_contains("compiling new PopIT function")

    # But the same shapes and dtypes will allow function reuse
    f(x.to("xla")).to("cpu")
    log = helpers.LogChecker(capfd)
    graph_hash_3, = log.findall(r"Graph hash is ([0-9]+)")
    log.assert_contains("reusing PopIT function")

    # And this graph hash will match the first graph...
    assert graph_hash_3 == graph_hash_1
    # but not the second, which has a different shape.
    assert graph_hash_3 != graph_hash_2


@pytest.mark.ipuHardwareRequired
def test_argument_lookup():
    import poptorch.eager as poptorch  # pylint: disable=unused-import, import-outside-toplevel
    poptorch.eager_options.use_lazy_tensor = True

    input = (torch.zeros(2, 3, 5).to('xla'), torch.zeros(2, 3, 5).to('xla'))
    poptorch.markStep()

    res = torch.add(*input)
    poptorch.markStep()

    out = torch.ones_like(res)
    poptorch.markStep()

    # Note this will be removed by dead code elimination leaving making the
    # compiled graph the same as for the original torch.add op
    out.clone()
    res2 = torch.add(*input)

    poptorch.markStep()

    helpers.assert_allequal(expected=res.cpu(), actual=res2.cpu())
