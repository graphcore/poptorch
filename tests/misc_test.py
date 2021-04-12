#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import pytest
import torch
import poptorch


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


def test_zero_size_tensor_error():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(x, size=(10, 10))

    x = torch.randn(0, 2, 5, 5)
    poptorch_model = poptorch.inferenceModel(Model())

    with pytest.raises(
            RuntimeError,
            match=
            r"Zero-sized tensors are unsupported \(Got shape \[0, 2, 5, 5\]\)"
    ):
        poptorch_model(x)
