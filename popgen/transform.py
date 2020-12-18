# Copyright (c) 2020 Graphcore Ltd. All rights reserved

from popgen import registry, Value
from popgen.values import InputValue


# generate_complex_ops(value)
#
# Apply simplification rules to the expression rooted at the parameter.
# New values are annotated with the applied transformation.
# Parameters:
#   value - root of an expression
# Returns: root af simplified expression
def generate_complex_ops(value):
    # munch(value, pattern)
    #
    # Attempt to match a pattern to the expression rooted at value
    # Parameters:
    #   value - root of the original expression
    #   pattern - root of the pattern
    # Returns: tuple(match, list([(idx, arg), ...])
    #   match - True / False according to whether matchig was successful
    #   (idx, arg) - arguments of the new complex operator
    #       idx - index of pattern's input node
    #       arg - value that is to become an argument with said index
    def munch(value, pattern):
        if isinstance(pattern, InputValue):
            return (True, [(pattern.num, value)])
        if not pattern.same(value):
            return (False, None)

        match = True
        new_args = []
        for i, _ in enumerate(pattern.args):
            (match, args) = munch(value.args[i], pattern.args[i])
            if not match:
                new_args = None
                break
            new_args += args

        return (match, new_args)

    # Attempt to match patterns in reverse order of weight and stop at
    # first match. Repeat process recursively.
    for name, op in registry.complex_ops.items():
        (_, pattern) = op
        (match, pos_args) = munch(value, pattern)
        if match:
            new_args = [None] * len(pos_args)
            for (pos, arg) in pos_args:
                new_args[pos] = arg

            new_value = Value(name, new_args)
            new_value.annotation = value.annotation
            new_value.annotate("// matched " + name + ": " + pattern.render())
            return generate_complex_ops(new_value)

    value.args = [generate_complex_ops(arg) for arg in value.args]
    return value


# value_numbering(value)
#
# Perform value numbering. Any identical values will be merged into a single.
# object. The tree rooted at the parameter becomes and acyclic graph.
# Parameters:
#   value - root of an expression tree
# Returns: potentially new root of an acyclic graph
def value_numbering(value):
    vn = dict()

    def numbered_value(value):
        for i, _ in enumerate(value.args):
            value.args[i] = numbered_value(value.args[i])

        key = value.vn()
        if key in vn:
            return vn[key]
        vn[key] = value
        return value

    return numbered_value(value)


# validate_forwarding(source):
#
# Ensure the forwarding of source is sane and resolve any chained rules by closure.
# Parameters:
#   source - name of operator being forwarded
def validate_forwarding(source):
    visited = set(source)
    dest = registry.forwardings[source]

    assert source not in registry.handlers, \
        source + " is both forwarded and handled"

    while dest not in registry.handlers:
        assert dest in registry.forwardings, \
            source + " forwarded but no handler found"
        assert dest not in visited, source + " has circular forwarding"
        visited.add(dest)
        dest = registry.forwardings[dest]

    registry.forwardings[source] = dest
