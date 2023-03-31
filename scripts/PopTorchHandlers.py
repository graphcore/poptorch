#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os

from popgen.api import expand, convert, generate
from popgen.helpers import cint, clong, cstr, tensor_list
from popgen.values import OriginalNode
from popgen.operatorfactory import op
from utils import _utils

script = "PopTorchHandlers.py"
output_dir = os.path.join(_utils.sources_dir(),
                          "poptorch/source/popart_canonicalization")

convert("recomputation_checkpoint", 1, "recomputationCheckpoint")
convert("update_param_inplace", 2, "copyvarupdate")

expand("begin_ipu_block", lambda x, y, z: op.beginIpuBlock(
    clong(x), clong(y), clong(z)))

expand("internal_cast", lambda tensor, dtype: op.internalCast(
    tensor, cstr(dtype)))
expand("call_cpu_op", lambda x, s: op.callCpuOp(tensor_list(x), cstr(s),
                                                OriginalNode()))
expand("identity_loss", lambda x, r: op.identityloss(x, cint(r)))
expand("optimizer_group", lambda x, l: op.optimizerGroup(
    clong(x), tensor_list(l)))
expand(
    "set_matmul_serialization", lambda x, s, a, b: op.setMatMulSerialization(
        x, cstr(s), clong(a), cint(b)))
expand("start_for_loop", op.startForLoop)
expand(
    "end_for_loop", lambda output, inputs, trip_count: op.endForLoop(
        output, inputs, clong(trip_count)))

expand("start_if_block", op.startIfBlock)
expand("start_else_block", op.startElseBlock)
expand("end_if_block", op.endIfBlock)

expand("nop", op.nop)

# These are graph annotations: they don't take any arguments and don't return
# anything: we just want to pass them through to the lowering stage.
expand("end_ipu_block", op.passThrough)
expand("begin_multi_conv", op.passThrough)
expand("pop_name_scope", op.passThrough)
expand("end_cpu_op", op.passThrough)

generate(script, "symbols::poptorch", output_dir + "/PoptorchHandlers.gen.cpp",
         globals())
