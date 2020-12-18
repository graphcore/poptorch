# Copyright (c) 2020 Graphcore Ltd. All rights reserved
import inspect
from popgen import generator, registry, values
from popgen.operatorfactory import op


# convert(aten, arity, popop=None, swizzles=None)
#
# Registers a conversion rule.
# Parameters:
#   aten - name of the operator to be converted
#   arity - number of inputs of aten
#   popop - popART operator to be generated (None: same as aten)
#   swizzles - list of integer indices representing a permutation of inputs
def convert(aten, arity, popop=None, swizzles=None):
    if popop is None:
        popop = aten

    if swizzles is None:
        swizzles = range(0, arity)

    inputs = []
    for swz in swizzles:
        assert isinstance(swz, int) and swz in range(0, arity), \
            "Illegal swizzle for " + aten
        inputs.append(values.InputValue("i" + str(swz), swz))

    fn = getattr(op, popop)
    registry.add_handler(aten, fn(*inputs), arity)


# expand(aten, fn)
#
# Registers an expansion rule
# Parametrs:
#   aten - name of operator to be expanded
#   fn - function defining the expansion
def expand(aten, fn):
    return registry.expand(aten, fn)


# forward(source, dest)
#
# Registers a forwarding rule. Effect is to forward one operator to the
# handlers of another.
# Parameters:
#   source - name of forwarded operator
#   dest - name of operator whoose handlers are to be used
def forward(source, dest):
    assert source not in registry.forwardings, \
        source + " is forwarded twice"
    registry.forwardings[source] = dest


# generate(namespace, filename)
#
# Generate C++ code.
# Parameters:
#   script - name of the top-level script
#   namespace - the namespace of the operators
#   filename - file to write the code to
#   global_symbols - dictionary of global_symbols from top-level
def generate(script, namespace, filename, global_symbols=globals()):
    generator.generate(script, namespace, filename, global_symbols)


# simplify(name, fn)
#
# Registers a simplification rule.
# Parameters:
#   operator_name - name of the operator to be greated as a string
#   fn - function defining the expression to be matched
def simplify(name, fn):
    # computes the weight of the expression (i.e. the number of values involved
    # in the pattern). The matched will use this to break ties - the heviest
    # pattern is preferred
    def weight(value):
        result = 1
        for arg in value.args:
            result += weight(arg)
        return result

    inputs = []
    ops = inspect.signature(fn).parameters
    for idx, op in enumerate(ops):
        inputs.append(values.InputValue(op, idx))

    pattern = fn(*inputs)
    registry.complex_ops[name] = (weight(pattern), pattern)
