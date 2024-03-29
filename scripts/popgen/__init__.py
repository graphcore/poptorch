# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import enum
import sys
from popgen import onnx

onnx.init()
onnx.parse_signatures()


class PtrOrRef(enum.Enum):
    PTR = 0
    REF = 1


# Root class for all expressions - the result of applying an operator
# to a list of arguments
class Value:
    def __init__(self, op, args, const=False, ptr_or_ref=None):
        assert isinstance(args, list), \
               "args should be a list in Value::__init__"

        self.op = op
        self.args = args
        self.cname = ""
        self.graph_arity = None
        self.annotation = []
        self.const = const
        self.ptr_or_ref = ptr_or_ref

        # perform dynamic casting for literals - makes for nice syntax
        for i, arg in enumerate(args):
            if isinstance(arg, float):
                self.args[i] = ConstantFloat(arg)

        # emit tensor parameters in an initilizer list
        self.tensor_braces = True

    # operator overloading - syntax sugar
    # note that we can't support __eq__ -- it would make the object unhashable
    def __add__(self, other):
        return Value('add', [self, other])

    def __ge__(self, other):
        return Value('logical_or', [self > other, self.equal(other)])

    def __gt__(self, other):
        return Value('greater', [self, other])

    def __le__(self, other):
        return Value('logical_or', [self < other, self.equal(other)])

    def __lt__(self, other):
        return Value('less', [self, other])

    def __mul__(self, other):
        return Value('mul', [self, other])

    def __ne__(self, other):
        return Value('logical_not', [self.equal(other)])

    def __neg__(self):
        return Value('neg', [self])

    def __sub__(self, other):
        return Value('sub', [self, other])

    def __truediv__(self, other):
        return Value('div', [self, other])

    def __radd__(self, other):
        return Value('add', [other, self])

    def __rmul__(self, other):
        return Value('mul', [other, self])

    def __rsub__(self, other):
        return Value('sub', [other, self])

    def __rtruediv__(self, other):
        return Value('div', [other, self])

    def equal(self, other):
        return Value('equal', [self, other])

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
    # Returns: index of the next available temp variable
    def emit(self, values, val_id, tabs, f=sys.stdout, root=False):
        if self in values:
            return val_id

        val_id = self.emit_arguments(values, val_id, tabs, f)
        self.emit_annotations(tabs, f)

        # split tensor and non-tensor arguments
        if not self.args or isinstance(self.args[0], NonTensorValue):
            tensors = []
            non_tensors = [values[arg] for arg in self.args]
            self.tensor_braces = False
        else:
            last_tensor = next(arg for arg in reversed(self.args)
                               if not isinstance(arg, NonTensorValue))
            last_tensor = len(self.args) - self.args[::-1].index(last_tensor)
            tensors = [values[arg] for arg in self.args[:last_tensor]]
            non_tensors = [values[arg] for arg in self.args[last_tensor:]]

        suffix = ";\n"
        if not root:
            suffix = "->output();\n"

        val_id = self.emit_assign_return(values,
                                         val_id,
                                         root,
                                         tabs,
                                         f,
                                         ptr_or_ref=PtrOrRef.PTR)
        left_brace = ["{"] if self.tensor_braces else []
        right_brace = ["}"] if self.tensor_braces else []

        if self.op is None:
            f.write("nullptr" + suffix)
        else:
            capital_op = self.op[0].upper() + self.op[1:]
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
    def emit_assign_return(self,
                           values,
                           val_id,
                           root,
                           tabs,
                           f,
                           const=False,
                           ptr_or_ref=None):
        if root:
            f.write(tabs + "return ")
            return val_id

        if isinstance(val_id, str):
            values[self] = val_id
        else:
            values[self] = "t" + str(val_id)
            val_id += 1

        pr_qual = ""
        if ptr_or_ref == PtrOrRef.PTR:
            pr_qual = "*"
        elif ptr_or_ref == PtrOrRef.REF:
            pr_qual = "&"

        const_qual = ""
        if const:
            const_qual = "const "

        f.write(tabs + const_qual + "auto " + pr_qual + values[self] + " = ")
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
        if self.op is None:
            return "<pass through>"
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

        if len(self.args) > 0:
            val_id = self.emit_arguments(values, val_id, tabs, f)
            val_id = self.emit_assign_return(values,
                                             val_id,
                                             root,
                                             tabs,
                                             f,
                                             ptr_or_ref=PtrOrRef.PTR)
            self.emit_call(
                "createConstantFloatLike",
                ["graph", values[self.args[0]], "{",
                 str(self.val), "}", "{}"], suffix, f)
        else:
            val_id = self.emit_assign_return(values,
                                             val_id,
                                             root,
                                             tabs,
                                             f,
                                             ptr_or_ref=PtrOrRef.PTR)
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
