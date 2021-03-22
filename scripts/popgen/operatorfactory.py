# Copyright (c) 2020 Graphcore Ltd. All rights reserved
from popgen import NonTensorValue, Value, onnx, poptorch
from popgen.helpers import empty_initializer


# no_tensor_braces(v):
#
# Modifiers for values that take tensors without initializer list braces
# Parameters:
#   v - the input value
def no_tensor_braces(v):
    v.tensor_braces = False
    return v


# def check_operator_signature(value, signatures)
#
# Verify an operator has correct signature
# Parameters:
#   value - the operator
#   signatures - signatures' dictionary
def check_operator_signature(value, signatures):
    assert value.op in signatures, \
        str(value.op) + " is not a supported operator"

    actual_args = value.args
    expected_args = signatures[value.op]

    # check non-tensor arguments
    first_non_tensor = -1
    if expected_args[0] == 'Args':
        for i, arg in enumerate(actual_args):
            if arg.op == 'empty_initializer':
                continue
            if isinstance(arg, NonTensorValue):
                first_non_tensor = i
                break

        assert first_non_tensor != 0, 'Expecting at least 1 tensor ' + \
            'argument for ' + value.op

    # no non-tensor arguments
    if first_non_tensor == -1:
        return value

    # check non-tensor arguments
    expected_args = expected_args[1:]
    actual_args = actual_args[first_non_tensor:]

    # assume any missing arguments are optional
    for i in range(1, len(expected_args) - len(actual_args)):
        actual_args.append('None')

    for i, arg in enumerate(actual_args):
        if isinstance(arg, Value):
            arg = arg.op
        assert arg in expected_args[i], 'Incorrect operand ' + str(i) + \
            'for ' + value.op + '. Got ' + arg + ' , expecting ' + \
            'one of: ' + str(expected_args[i])

    return value


# Factory class for creating popArt ops. Operators are created
# on the fly based on spelling of attributes.
class OperatorFactory:
    def __getattr__(self, name):
        if name in onnx.signatures:
            return lambda *args: \
                check_operator_signature(Value(name, list(args)), \
                onnx.signatures)
        if name in poptorch.signatures:
            return lambda *args: \
                check_operator_signature(Value(name, list(args)), \
                poptorch.signatures)
        raise ValueError(name + " is not a supported operator")

    def cast(self, t, ty):
        value = no_tensor_braces(Value('cast', [t, ty]))
        check_operator_signature(value, poptorch.signatures)
        return value

    def internalCast(self, t, ty):
        value = no_tensor_braces(Value('internalCast', [t, ty]))
        check_operator_signature(value, poptorch.signatures)
        return value

    def constantPad(self, x, l, c):
        value = no_tensor_braces(Value('constantPad', [x, l, c]))
        check_operator_signature(value, poptorch.signatures)
        return value

    def edgePad(self, t, l):
        value = no_tensor_braces(Value('edgePad', [t, l]))
        check_operator_signature(value, poptorch.signatures)
        return value

    def printIpuTensor(self, t, s):
        value = no_tensor_braces(Value('printIpuTensor', [t, s]))
        check_operator_signature(value, poptorch.signatures)
        return value

    def transpose(self, t):
        value = Value('transpose', [t, empty_initializer()])
        check_operator_signature(value, onnx.signatures)
        return value

    def randomNormal(self, x, shape, high, low, scalar_type=None):
        args = [x, shape, high, low]
        if scalar_type is not None:
            args += [scalar_type]

        value = Value('randomNormal', args)
        check_operator_signature(value, poptorch.signatures)
        return value

    def randomUniform(self, x, shape, high, low, scalar_type=None):
        args = [x, shape, high, low]
        if scalar_type is not None:
            args += [scalar_type]

        value = no_tensor_braces(Value('randomUniform', args))
        check_operator_signature(value, poptorch.signatures)
        return value

    def recomputationCheckpoint(self, x):
        value = no_tensor_braces(Value('recomputationCheckpoint', [x]))
        check_operator_signature(value, poptorch.signatures)
        return value

    def reflectionPad(self, t, l):
        value = no_tensor_braces(Value('reflectionPad', [t, l]))
        check_operator_signature(value, poptorch.signatures)
        return value

    def setAvailableMemory(self, x, y):
        value = no_tensor_braces(Value('setAvailableMemory', [x, y]))
        check_operator_signature(value, poptorch.signatures)
        return value

    def setMatMulSerialization(self, x, s, a, b):
        value = no_tensor_braces(Value('setMatMulSerialization', [x, s, a, b]))
        check_operator_signature(value, poptorch.signatures)
        return value

    def endForLoop(self, output, inputs, trip_count):
        value = no_tensor_braces(
            Value('endForLoop', [output, inputs, trip_count]))
        check_operator_signature(value, poptorch.signatures)
        return value


op = OperatorFactory()
