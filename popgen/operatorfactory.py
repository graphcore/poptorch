# Copyright (c) 2020 Graphcore Ltd. All rights reserved
from popgen import Value
from popgen.helpers import empty_initializer


# no_tensor_braces(v):
#
# Modifiers for values that take tensors without initializer list braces
# Parameters:
#   v - the input value
def no_tensor_braces(v):
    v.tensor_braces = False
    return v


# Factory class for creating popArt ops. Operators are created
# on the fly based on spelling of attributes.
class OperatorFactory:
    def __getattr__(self, name):
        return lambda *args: Value(name, list(args))

    def cast(self, t, ty):
        return no_tensor_braces(Value('cast', [t, ty]))

    def constantPad(self, x, l, c):
        return no_tensor_braces(Value('constantPad', [x, l, c]))

    def edgePad(self, t, l):
        return no_tensor_braces(Value('edgePad', [t, l]))

    def printIpuTensor(self, t, s):
        return no_tensor_braces(Value('printIpuTensor', [t, s]))

    def transpose(self, t):
        return Value('transpose', [t, empty_initializer()])

    def randomUniform(self, x, shape, high, low, scalar_type=None):
        args = [x, shape, high, low]
        if scalar_type is not None:
            args += [scalar_type]

        return no_tensor_braces(Value('randomUniform', args))

    def recomputationCheckpoint(self, x):
        return no_tensor_braces(Value('recomputationCheckpoint', [x]))

    def reflectionPad(self, t, l):
        return no_tensor_braces(Value('reflectionPad', [t, l]))

    def setAvailableMemory(self, x, y):
        return no_tensor_braces(Value('setAvailableMemory', [x, y]))

    def setMatMulSerialization(self, x, s, a, b):
        return no_tensor_braces(Value('setMatMulSerialization', [x, s, a, b]))


op = OperatorFactory()
