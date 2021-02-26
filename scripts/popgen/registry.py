# Copyright (c) 2020 Graphcore Ltd. All rights reserved
import inspect
import re
from popgen import values

# simplification rules. operator_name -> value
complex_ops = dict()

# forwardings. from_operator -> to_operator
forwardings = dict()

# operator handlers. operator_name -> list(value)
handlers = dict()


# add_handler(aten, value, arity)
#
# Register a new handler for an operator
# Parameters:
#   aten - name of operator
#   value - root of the expansion expression
#   arity - number of unique graph nodes taken as input
def add_handler(aten, value, arity):
    if aten not in handlers:
        handlers[aten] = []
    value.set_graph_arity(arity)
    handlers[aten].append(value)


# add_implicit_handlers(global_symbols)
#
# Inspect global namespace dictionary and register function handlers
# Parameters:
#   global_symbols - dictianary of top-level globals
def add_implicit_handlers(global_symbols):
    for name in global_symbols.keys():
        fn = global_symbols[name]
        if not callable(fn):
            continue

        res = re.search('(.+)_handler$', name)
        if res:
            expand(res.group(1), fn)


# clear(clear_complex_ops = False)
#
# Clears all internal dictionaries.
# Parameters:
#   clear_complex_ops - clear complex_ops map (default: False)
def clear(clear_complex_ops=False):
    handlers.clear()
    forwardings.clear()
    if clear_complex_ops:
        complex_ops.clear()


# expand(aten, fn)
#
# Registers an expansion rule
# Parametrs:
#   aten - name of operator to be expanded
#   fn - function defining the expansion
def expand(aten, fn):
    inputs = []
    ops = inspect.signature(fn).parameters
    for idx, op in enumerate(ops):
        inputs.append(values.InputValue(op, idx))

    add_handler(aten, fn(*inputs), len(ops))
