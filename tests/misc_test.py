#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import pytest
import torch
import torch.nn as nn
import poptorch
import helpers


@helpers.overridePoptorchLogLevel()
def test_set_log_level():
    for i in range(5):
        poptorch.setLogLevel(i)

    with pytest.raises(ValueError, match="Invalid log level integer"):
        poptorch.setLogLevel(5)

    poptorch.setLogLevel("TRACE")
    poptorch.setLogLevel("DEBUG")
    poptorch.setLogLevel("INFO")
    poptorch.setLogLevel("WARN")
    poptorch.setLogLevel("ERR")
    poptorch.setLogLevel("OFF")

    err_str = "Unknown log level: wibble. Valid values are DEBUG, ERR, INFO, "
    err_str += "OFF, TRACE and WARN"

    with pytest.raises(ValueError, match=err_str):
        poptorch.setLogLevel("wibble")


@helpers.printCapfdOnExit
@helpers.overridePopartLogLevel()
@pytest.mark.parametrize("trace_model", [True, False])
def test_set_popart_log_level(capfd, trace_model):
    # Only strings are allowed
    with pytest.raises(ValueError, match="Level must be one of"):
        poptorch._logging.setPopartLogLevel(0)  # pylint: disable=protected-access

    # Only some strings are allowed
    with pytest.raises(ValueError, match="Level must be one of"):
        poptorch._logging.setPopartLogLevel("FOO")  # pylint: disable=protected-access

    poptorch._logging.setPopartLogLevel("DEBUG")  # pylint: disable=protected-access
    poptorch._logging.setPopartLogLevel("INFO")  # pylint: disable=protected-access
    poptorch._logging.setPopartLogLevel("WARN")  # pylint: disable=protected-access

    model = torch.nn.Linear(2, 2)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)
    inference_model(torch.randn([2, 2]))

    log = helpers.LogChecker(capfd)
    log.assert_no_matches(r"popart:devicex \d+\.\d+ T:")
    log.assert_no_matches(r"popart:ir \d+\.\d+ D:")
    log.assert_no_matches(r"popart:ir \d+\.\d+ I:")
    log.assert_no_matches(r"popart:session \d+\.\d+ T:")
    log.assert_no_matches(r"popart:popart \d+\.\d+ T:")

    poptorch._logging.setPopartLogLevel("ERR")  # pylint: disable=protected-access
    poptorch._logging.setPopartLogLevel("OFF")  # pylint: disable=protected-access
    poptorch._logging.setPopartLogLevel("TRACE")  # pylint: disable=protected-access

    inference_model = poptorch.inferenceModel(model, options)
    inference_model(torch.randn([2, 2]))

    log = helpers.LogChecker(capfd)
    log.assert_matches(r"popart:devicex \d+\.\d+ T:")
    log.assert_matches(r"popart:ir \d+\.\d+ D:")
    log.assert_matches(r"popart:ir \d+\.\d+ I:")
    log.assert_matches(r"popart:session \d+\.\d+ T:")
    log.assert_matches(r"popart:popart \d+\.\d+ T:")


@pytest.mark.parametrize("trace_model", [True, False])
def test_zero_size_tensor_error(trace_model):
    class Model(torch.nn.Module):
        def forward(self, x):
            # The operation doesn't matter, we just want to produce the
            # failure on an operation that works with zero-sized tensors
            # in native Torch
            return torch.nn.functional.interpolate(x, size=(10, 10))

    x = torch.randn(0, 2, 5, 5)
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(Model(), options)

    with pytest.raises(
            poptorch.Error,
            match=
            r"Zero-sized tensors are unsupported \(Got shape \[0, 2, 5, 5\]\)"
    ):
        poptorch_model(x)


orig_input_trace_tensors = [
    torch.tensor([1.], dtype=torch.half),
    torch.tensor([2.]),
    torch.tensor([3], dtype=torch.int32)
]


def print_orig_input_trace_harness(trace_model, capfd, model, orig_types,
                                   *input_args):
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    inference_model = poptorch.inferenceModel(model, options)
    inference_model(*input_args)

    testlog = helpers.LogChecker(capfd)

    orig_str = "[orig:" + orig_types + "]"
    trace_types = orig_types.replace("Half", "Float")
    trace_str = "graph(" + trace_types + "):\n"
    testlog.assert_contains(trace_str + orig_str)


@helpers.overridePoptorchLogLevel("TRACE")
@helpers.printCapfdOnExit
@pytest.mark.parametrize("trace_model", [True, False])
def test_print_orig_input_trace_nested_tuple_tensors(capfd, trace_model):
    class Model(torch.nn.Module):
        def forward(self, xss):
            return xss[0][0] + xss[0][1] + xss[1][0]

    print_orig_input_trace_harness(
        trace_model, capfd, Model(),
        "%xss : ((Half(1, strides=[1], requires_grad=0, device=cpu), " +
        "Float(1, strides=[1], requires_grad=0, device=cpu)), " +
        "(Int(1, strides=[1], requires_grad=0, device=cpu)))",
        ((orig_input_trace_tensors[0], orig_input_trace_tensors[1]),
         (orig_input_trace_tensors[2], )))


@helpers.overridePoptorchLogLevel("TRACE")
@helpers.printCapfdOnExit
@pytest.mark.parametrize("trace_model", [True, False])
def test_print_orig_input_trace_tuple_tensors(capfd, trace_model):
    class Model(torch.nn.Module):
        def forward(self, xs):
            return xs[0] + xs[1] + xs[2]

    print_orig_input_trace_harness(
        trace_model, capfd, Model(),
        "%xs : (Half(1, strides=[1], requires_grad=0, device=cpu), " +
        "Float(1, strides=[1], requires_grad=0, device=cpu), " +
        "Int(1, strides=[1], requires_grad=0, device=cpu))",
        (orig_input_trace_tensors[0], orig_input_trace_tensors[1],
         orig_input_trace_tensors[2]))


@helpers.overridePoptorchLogLevel("TRACE")
@helpers.printCapfdOnExit
@pytest.mark.parametrize("trace_model", [True, False])
def test_print_orig_input_trace_tensors(capfd, trace_model):
    class Model(torch.nn.Module):
        def forward(self, x, y, z):
            return x + y + z

    print_orig_input_trace_harness(
        trace_model, capfd, Model(),
        "%x : Half(1, strides=[1], requires_grad=0, device=cpu),\n      " +
        "%y : Float(1, strides=[1], requires_grad=0, device=cpu),\n      " +
        "%z : Int(1, strides=[1], requires_grad=0, device=cpu)",
        orig_input_trace_tensors[0], orig_input_trace_tensors[1],
        orig_input_trace_tensors[2])


@pytest.mark.parametrize("trace_model", [True, False])
def test_untracable_type_error(trace_model):
    class Model(torch.nn.Module):
        def forward(self, t, f):
            return t + torch.tensor([f])

    x = torch.tensor([3.4])
    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(Model(), options)

    with pytest.raises(
            TypeError,
            match=
            r"All forward function arguments used to compile and run the model "
            r"must be Tensors or \(possibly nested\) Lists and Tuples of "
            r"Tensors \(Got types: \[Tensor, float\]\)."):
        poptorch_model(x, 5.6)


def test_torch_backward_error():
    x = torch.Tensor([5.0])
    model = helpers.ModelWithWeights(lambda x: x, x.shape)
    poptorch_model = poptorch.trainingModel(model)
    poptorch_out, poptorch_loss = poptorch_model((x, ))

    error_message = (
        r"backward\(\) cannot be called explicitly on "
        r"outputs of a PopTorch model. If you're using a trainingModel, "
        r"the backwards pass is performed automatically when invoking the "
        r"model. If you're using an inferenceModel, you should use a "
        r"trainingModel instead.")

    with pytest.raises(poptorch.Error, match=error_message):
        poptorch_out.backward()
    with pytest.raises(poptorch.Error, match=error_message):
        poptorch_loss.backward()


@pytest.mark.parametrize(
    "error_type", poptorch.poptorch_core.TestErrorType.__members__.values())
def test_generic_error_handling(error_type):
    with pytest.raises(poptorch.Error) as e:
        poptorch.poptorch_core._throwTestError(error_type)  # pylint: disable=protected-access
    assert "throwTestError::bottomLevel" in e.value.args[0]
    assert "throwTestError::topLevel" in e.value.args[0]


def test_specific_error_handling():
    try:
        poptorch.poptorch_core._throwTestError(  # pylint: disable=protected-access
            poptorch.poptorch_core.TestErrorType.PoplarRecoverableFullReset)
        assert False, "Expected an error to be thrown"
    except poptorch.RecoverableError as e:
        assert e.recovery_action == "FULL_RESET"
        assert "throwTestError::bottomLevel" in e.location
        assert "throwTestError::topLevel" in e.location
        assert e.type == "poplar_recoverable_runtime_error"
        # Message shouldn't contain any backtrace
        assert "throwTestError::bottomLevel" not in e.message
        assert "throwTestError::topLevel" not in e.message

    try:
        poptorch.poptorch_core._throwTestError(  # pylint: disable=protected-access
            poptorch.poptorch_core.TestErrorType.PoplarLinkError)
        assert False, "Expected an error to be thrown"
    except poptorch.Error as e:
        # Make sure the backtrace was reset between the two exceptions
        assert e.location.count("throwTestError::bottomLevel") == 1
        assert e.location.count("throwTestError::topLevel") == 1
        assert e.type == "poplar_link_error"
        # Message shouldn't contain any backtrace
        assert "throwTestError::bottomLevel" not in e.message
        assert "throwTestError::topLevel" not in e.message

        # Make sure the link error is added at the end of the error message
        assert "-lfoo not found" in e.message

    try:
        poptorch.poptorch_core._throwTestError(  # pylint: disable=protected-access
            poptorch.poptorch_core.TestErrorType.PoplarUnrecoverable)
        assert False, "Expected an error to be thrown"
    except poptorch.UnrecoverableError as e:
        # Make sure the backtrace was reset between the two exceptions
        assert e.location.count("throwTestError::bottomLevel") == 1
        assert e.location.count("throwTestError::topLevel") == 1
        assert e.type == "poplar_unrecoverable_runtime_error"
        # Message shouldn't contain any backtrace
        assert "throwTestError::bottomLevel" not in e.message
        assert "throwTestError::topLevel" not in e.message


@helpers.printCapfdOnExit
@helpers.overridePopartLogLevel("DEBUG")
@pytest.mark.parametrize("trace_model", [True, False])
def test_outline_attribute(capfd, trace_model):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gn1 = torch.nn.GroupNorm(4, 8)
            self.gn2 = torch.nn.GroupNorm(2, 8)

        def forward(self, x):
            with poptorch.Attribute(__outline={"layer": "embedding"}):
                x = self.gn1(x)
            return self.gn2(x)

    input = torch.randn(3, 8)

    options = poptorch.Options()
    options.Jit.traceModel(trace_model)
    poptorch_model = poptorch.inferenceModel(Model(), options)

    poptorch_model(input)

    testlog = helpers.LogChecker(capfd)

    get_regex = lambda op_name: (f'Op "{op_name}/GroupNormalization", '
                                 r"[0-9]+ of type ai\.graphcore\."
                                 "GroupNormalization:1"
                                 r"(?:\n.+)+"
                                 f"{op_name}"
                                 r".+(?:\n.+)+"
                                 "layer: layer:embedding")

    # Ensure the first group norm has the outline attribute
    testlog.assert_matches(get_regex("gn1"), per_line=False)

    # Ensure the second group norm doesn't have the attribute,
    # as it is outside the attribute scope
    testlog.assert_no_matches(get_regex("gn2"), per_line=False)


# Note: the ipu models are not supported by poptorch.ConnectionType.Never
@pytest.mark.ipuHardwareRequired
def test_compile_without_ipu():
    class SimpleAdder(nn.Module):
        def forward(self, x, y):
            return x + y

    model = SimpleAdder()
    opts = poptorch.Options().connectionType(poptorch.ConnectionType.Never)
    inference_model = poptorch.inferenceModel(model, opts)

    t1 = torch.tensor([1.])
    t2 = torch.tensor([2.])

    inference_model.compile(t1, t2)
