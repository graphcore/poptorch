#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from popgen.api import expand, convert, generate
from popgen.helpers import cfloat, cint, clong, cstr, tensor_list
from popgen.operatorfactory import op

script = "PopTorchHandlers.py"
output_dir = "poptorch/source/popart_canonicalization"

convert("recomputation_checkpoint", 1, "recomputationCheckpoint")

expand("begin_ipu_block", lambda x, y, z: op.beginIpuBlock(
    clong(x), clong(y), clong(z)))
expand("ipu_print_tensor", lambda x, s: op.printIpuTensor(x, cstr(s)))
expand("identity_loss", lambda x, r: op.identityloss(x, cint(r)))
expand("optimizer_group", lambda x, l: op.optimizerGroup(
    clong(x), tensor_list(l)))
expand(
    "set_available_memory", lambda x, y: op.setAvailableMemory(x, cfloat(y)))
expand(
    "set_matmul_serialization", lambda x, s, a, b: op.setMatMulSerialization(
        x, cstr(s), clong(a), cint(b)))

generate(script, "symbols::poptorch", output_dir + "/PoptorchHandlers.gen.cpp",
         globals())
