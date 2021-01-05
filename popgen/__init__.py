# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import sys
from popgen import onnx

onnx.init()
onnx.parse_signatures()


# Root class for all expressions - the result of applying an operator
# to a list of arguments
class Value:
    def __init__(self, op, args):
        assert isinstance(args, list), \
               "args should be a list in Value::__init__"

        self.op = op
        self.args = args
        self.cname = ""
        self.graph_arity = None
        self.annotation = []

        # perform dynamic casting for literals - makes for nice syntax
        for i, arg in enumerate(args):
            if isinstance(arg, float):
                self.args[i] = ConstantFloat(arg)

        # emit tensor parameters in an initilizer list
        self.tensor_braces = True

    # operator overloading - syntax sugar
    def __add__(self, other):
        return Value('add', [self, other])

    def __gt__(self, other):
        return Value('greater', [self, other])

    def __mul__(self, other):
        return Value('mul', [self, other])

    def __sub__(self, other):
        return Value('sub', [self, other])

    def __truediv__(self, other):
        return Value('div', [self, other])

    def __radd__(self, other):
        return Value('add', [other, self])

    def __rsub__(self, other):
        return Value('sub', [other, self])

    def __rtruediv__(self, other):
        return Value('div', [other, self])

    def set_graph_arity(self, arity):
        self.graph_arity = arity

    def annotate(self, annot):
        self.annotation.append(annot)

    # emit(values, val_id, tabs, f, root)
    #
    # Emits C++ code for this value
    # Parameters:
    #   values - map of previously generated Value objects and their C++ images (Value -> string)
    #   val_id - the index of the first available temp variable
    #   tabs - indentation string
    #   f - output stream
    #   root - True: we should generate a return statement
    # Returns: index of the next available temp  variable
    def emit(self, values, val_id, tabs, f=sys.stdout, root=False):
        if self in values:
            return val_id

        val_id = self.emit_arguments(values, val_id, tabs, f)
        self.emit_annotations(tabs, f)

        # split tensor and non-tensor arguments
        if isinstance(self.args[0], NonTensorValue):
            tensors = []
            non_tensors = [values[arg] for arg in self.args]
            self.tensor_braces = False
        else:
            last_tensor = next(arg for arg in reversed(self.args)
                               if not isinstance(arg, NonTensorValue))
            last_tensor = len(self.args) - self.args[::-1].index(last_tensor)
            tensors = [values[arg] for arg in self.args[:last_tensor]]
            non_tensors = [values[arg] for arg in self.args[last_tensor:]]

        capital_op = self.op[0].upper() + self.op[1:]
        suffix = ";\n"
        if not root:
            suffix = "->output();\n"

        val_id = self.emit_assign_return(values, val_id, root, tabs, f)
        left_brace = ["{"] if self.tensor_braces else []
        right_brace = ["}"] if self.tensor_braces else []

        self.emit_call("create" + capital_op, ["graph"] + left_brace +
                       tensors + right_brace + non_tensors, suffix, f)

        return val_id

    # emit_arguments(values, val_id, tabs, f)
    #
    # Emits C++ code for the arguments this value
    # Parameters:
    #   values - map of previously generated Value objects and their C++ images (Value -> string)
    #   val_id - the index of the first available temp variable
    #   tabs - indentation string
    #   f - output stream
    # Returns: index of the next available temp  variable
    def emit_arguments(self, values, val_id, tabs, f):
        for arg in self.args:
            val_id = arg.emit(values, val_id, tabs, f, False)
        return val_id

    # emit_annotations(tabs, f)
    #
    # Emits annotations as C++ comments
    # Parameters:
    #   tabs - indentation string
    #   f - output stream
    def emit_annotations(self, tabs, f):
        for annot in self.annotation:
            f.write(tabs + annot + "\n")

    # emit_assign_return(values, val_id, root, tabs, f)
    #
    # Emits either an assignment or a return statement
    # Parameters:
    #   values - map of previously generated Value objects and their C++ images (Value -> string)
    #   val_id - the index of the first available temp variable
    #   tabs - indentation string
    #   f - output stream
    # Returns: index of the next available temp  variable
    def emit_assign_return(self, values, val_id, root, tabs, f):
        if root:
            f.write(tabs + "return ")
            return val_id

        if isinstance(val_id, str):
            values[self] = val_id
        else:
            values[self] = "t" + str(val_id)
            val_id += 1

        f.write(tabs + "auto " + values[self] + " = ")
        return val_id

    # emit_call(fname, args, suffix, f)
    #
    # Emit a function call
    # Parameters:
    #   fname - function name
    #   args - arguments as list of strings
    #   suffix - string to prepend after call
    #   f - output stream
    def emit_call(self, fname, args, suffix, f):
        f.write(fname + "(")
        for (i, arg) in enumerate(args):
            if i > 0:
                if arg not in ["}"] and args[i - 1] not in ["{"]:
                    f.write(", ")
            f.write(arg)
        f.write(')' + suffix)

    # vn()
    #
    # Return a value number for this object
    # Returns: tuple(operator, value numbers of arguments)
    def vn(self):
        return tuple([self.op] + [arg.vn() for arg in self.args])

    # same(other)
    #
    # Returns True if the other operator is a potential match for this one
    def same(self, other):
        return self.op == other.op

    # render()
    #
    # Returns a string image of this object. Used for C++ annotations.
    def render(self):
        string = self.op + '('
        for i, arg in enumerate(self.args):
            if i > 0:
                string += ', '
            string += arg.render()
        return string + ')'


# ConstantFloat(val)
#
# Represents a constant floating point value to be used as tensor argument
# Parameters:
#   val - the floating point constant
class ConstantFloat(Value):
    def __init__(self, val):
        Value.__init__(self, 'float', [])
        self.val = val

    def emit(self, values, val_id, tabs, f=sys.stdout, root=False):
        if self in values:
            return val_id

        suffix = ";\n"
        if not root:
            suffix = "->output();\n"

        val_id = self.emit_assign_return(values, val_id, root, tabs, f)
        if len(self.args) > 0:
            self.emit_call(
                "createConstantFloatLike",
                ["graph", values[self.args[0]], "{",
                 str(self.val), "}", "{}"], suffix, f)
        else:
            self.emit_call(
                "createConstantFloat32",
                ["graph", "{", str(self.val), "}", "{}"], suffix, f)
        return val_id

    def vn(self):
        return str(self.val)

    def same(self, other):
        return other.op == 'float' and self.val == other.val

    def render(self):
        return str(self.val)


# NonTensorValue(op, args)
#
# Root class for non-tensor values.
# Parameters:
#   op - operator
#   args - arguments
class NonTensorValue(Value):
    def __init__(self, op, args):
        Value.__init__(self, op, args)
