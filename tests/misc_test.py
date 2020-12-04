#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import poptorch
import pytest


def test_setLogLevel():
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
