# Copyright (c) 2020 Graphcore Ltd. All rights reserved

# signatures for manually added operators
signatures = {
    'beginIpuBlock': [['clong'], ['clong'], ['clong']],
    'cast': ['Args', ['scalar_type']],
    'internalCast': ['Args', ['cstr']],
    'constantPad': ['Args', ['clong_list'], ['cfloat']],
    'edgePad': ['Args', ['clong_list']],
    'optimizerGroup': [['clong'], ['tensor_list']],
    'printIpuTensor': ['Args', ['cstr']],
    'callCpuOp': [['tensor_list'], ['cstr'], ['node']],
    'randomNormal': [
        'Args', ['tensor_shape'], ['cfloat'], ['cfloat'],
        ['scalar_type', 'None']
    ],
    'randomUniform': [
        'Args', ['tensor_shape'], ['cfloat'], ['cfloat'],
        ['scalar_type', 'None']
    ],
    'recomputationCheckpoint': ['Args'],
    'reflectionPad': ['Args', ['clong_list']],
    'setAvailableMemory': ['Args', ['cfloat']],
    'setMatMulSerialization': ['Args', ['cstr'], ['clong'], ['cint']],
    'startForLoop': ['Args'],
    'endForLoop': ['Args', ['clong']],
}
