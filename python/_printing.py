# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch


# Override torches repr function to provide information on the pre hooks as
# well. The pre hooks is where BeginBlock is added
def module_repr(m: torch.nn.Module):
    """
    Provide a string representation of a torch.nn.Module along with the
    corresponding pre-hooks.

    This will show any BeginBlocks that have been added to the model which
    otherwise wouldn't be displayed.
    """

    def _add_indent(s_, numSpaces):
        return f'\n{numSpaces}'.join(s_.split('\n'))

    # pylint: disable=protected-access

    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []
    extra_repr = m.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []
    for key, module in m._modules.items():
        mod_str = module_repr(module)
        mod_str = _add_indent(mod_str, 2)
        child_lines.append('(' + key + '): ' + mod_str)
    lines = extra_lines + child_lines

    pre_hooks = ''.join(
        map(lambda x: repr(x) + ' ', m._forward_pre_hooks.values()))

    main_str = pre_hooks + m._get_name() + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str


def print(m: torch.nn.Module):
    """
    Prints a torch.nn.Module along with the corresponding pre-hooks.

    This will print any BeginBlocks that have been added to the model which
    otherwise wouldn't be displayed.
    """
    print(module_repr(m))
