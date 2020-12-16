#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from popgen.api import expand, generate, op
from popgen.helpers import cstr

script = "PopTorchHandlers.py"
output_dir = "poptorch/source/popart_canonicalization"

# poptorch handlers in a different file
expand("ipu_print_tensor", lambda x, s: op.printIpuTensor(x, cstr(s)))

generate(script, "symbols::poptorch", output_dir + "/PoptorchHandlers.gen.cpp",
         globals())
