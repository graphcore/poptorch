# Copyright (c) 2020 Graphcore Ltd. All rights reserved

import sys
from popgen import Value, NonTensorValue, op


# AlphaValue(args)
#
# Represents the alpha computation required for operators that perform
# implicit scaling. Its purpose is to avoid a multiplication by unity.
# Parameters:
#   args[0] - value to be scaled
#   args[1] - scaling factor
class AlphaValue(Value):
    def __init__(self, args):
        Value.__init__(
            self, 'alpha',
            [args[0], args[1], op.mul(args[0], args[1])])

    def emit(self, values, val_id, tabs, f=sys.stdout, root=False):
        if self in values:
            return val_id

        val_id = self.emit_arguments(values, val_id, tabs, f)
        self.emit_annotations(tabs, f)
        values[self] = "t" + str(val_id)
        f.write(tabs + "auto " + values[self] + " = hasUnityValue(" +
                values[self.args[1]] + ") ? " + values[self.args[0]] + " : " +
                values[self.args[2]] + ";\n")
        return val_id + 1


# CastInPlace(op, args, to_type)
#
# Represents the operation of swiching the scalar type of a tensor.
# It is a cast "in-place" in the sense that it doesn't generate
# any casting nodes.
# Parameters:
#   op - name of operator
#   args - tensor input
#   to_type - name of target type
class CastInPlace(Value):
    def __init__(self, op, args, to_type):
        Value.__init__(self, op, args)
        self.to_type = to_type

    def emit(self, values, val_id, tabs, f=sys.stdout, root=False):
        if self in values:
            return val_id
        val_id = self.emit_arguments(values, val_id, tabs, f)
        self.emit_annotations(tabs, f)

        node = "t" + str(val_id)
        f.write(tabs + "auto " + node + " = " + values[self.args[0]] +
                "->node();\n")
        f.write(tabs + node + "->t_(c10::attr::value,\n")
        f.write(tabs + "     " + " " * len(node) + node +
                "->t(c10::attr::value).to(" + self.to_type + "));\n")
        f.write(tabs + node + "->output()->inferTypeFrom(" + node +
                "->t(c10::attr::value));\n")

        if not root:
            values[self] = "t" + str(val_id + 1)
            f.write(tabs + "auto " + values[self] + " = " + node +
                    "->output();\n")
            return val_id + 2
        f.write(tabs + "return " + node + "->output();\n")
        return val_id + 1


# Helper(op, args, method, expects_node)
#
# A wrapper class for helper methods that return tensors
# Parameters:
#   op - operator
#   args - arguments
#   method - generation method
#   expects_node - True if arguments should be typed Node* instead of Value*
class Helper(Value):
    def __init__(self, op, args, method, expects_node=False):
        Value.__init__(self, op, args)
        self.method = method
        self.expects_node = expects_node

    def emit(self, values, val_id, tabs, f=sys.stdout, root=False):
        if self in values:
            return val_id

        val_id = self.emit_arguments(values, val_id, tabs, f)
        self.emit_annotations(tabs, f)

        args = [values[arg] for arg in self.args]
        if self.expects_node:
            args = [arg + "->node()" for arg in args]

        val_id = self.emit_assign_return(values, val_id, root, tabs, f)
        self.emit_call(self.method, args, ";\n", f)
        return val_id


# InputValue(name, num)
#
# Represents an input to an operator
# Parameters:
#   name - name of input
#   num - index of input
class InputValue(Value):
    def __init__(self, name, num):
        Value.__init__(self, 'input', [])
        self.name = name
        self.num = num

    def emit(self, values, val_id, tabs, f=sys.stdout, root=False):
        assert not root, "input values cannot be root expression"
        if self in values:
            return val_id

        self.emit_assign_return(values, self.name, root, tabs, f)
        f.write("node->input(" + str(self.num) + ");\n")
        return val_id

    def vn(self):
        return self.name

    def same(self, other):
        return True

    def render(self):
        return self.name


# OutputValue(index)
#
# Represents the output value of an operator. This is useful for
# occasions where we need the expected shape of the output.
# Parameters:
#   index - index of output
class OutputValue(Value):
    def __init__(self, index):
        Value.__init__(self, 'output' + str(index), [])
        self.index = index

    def emit(self, values, val_id, tabs, f=sys.stdout, root=False):
        assert not root, "output values may not be root expressions"
        if self in values:
            return val_id

        val_id = self.emit_assign_return(values, val_id, root, tabs, f)
        f.write("node->output(" + str(self.index) + ");\n")
        return val_id + 1


# NonTensorConstant(op, val, method)
#
# Represents a constant value that is not a tensor. Supports literals
# as well as graph constants.
# Parameters:
#   val - the constant value
#   method - helper method to be called when the value is not a literal
class NonTensorConstant(NonTensorValue):
    def __init__(self, op, val, method):
        self.val = val
        self.method = method
        if isinstance(val, Value):
            NonTensorValue.__init__(self, op, [val])
        else:
            NonTensorValue.__init__(self, op, [])

    def emit(self, values, val_id, tabs, f=sys.stdout, root=False):
        assert not root, op + " cannot be a root expression"
        if not isinstance(self.val, Value):
            values[self] = str(self.val)
            return val_id
        if self in values:
            return val_id

        val_id = self.emit_arguments(values, val_id, tabs, f)
        self.emit_annotations(tabs, f)

        val_id = self.emit_assign_return(values, val_id, root, tabs, f)
        self.emit_call(self.method, [values[self.val] + "->node()"], ";\n", f)
        return val_id

    def vn(self):
        if isinstance(self.val, Value):
            return Value.vn(self)
        return str(self.val)

    def render(self):
        if isinstance(self.val, Value):
            return self.op + "(" + self.val.render() + ")"
        return str(self.val)

    def same(self, other):
        if self.op != other.op or len(self.args) != len(other.args):
            return False
        if isinstance(self.val, Value):
            return self.val.same(other.val)
        return self.render() == other.render()


# NonTensorHelper(op, args, method, expects_node)
#
# A wrapper class for helper methods that do not return tensors
# Parameters:
#   op - operator
#   args - arguments
#   method - generation method
#   expects_node - True if arguments should be typed Node* instead of Value*
class NonTensorHelper(NonTensorValue):
    def __init__(self, op, args, method, expects_node=False):
        NonTensorValue.__init__(self, op, args, expects_node)
        self.method = method

    def emit(self, values, val_id, tabs, f=sys.stdout, root=False):
        assert not root, op + " helper cannot be a root expression"
        return Helper.emit(self, values, val_id, tabs, f, False)
