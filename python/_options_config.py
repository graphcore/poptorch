#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import poptorch


def parseAndSetOptions(options, filepath):
    cmds = []
    with open(filepath) as f:
        filename = os.path.basename(f.name)
        prefix = "options."
        for line in f:
            # Remove whitespace
            stripped = line.strip()
            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                continue
            cmd = prefix + stripped
            cmds.append(cmd)

    code = "\n".join(cmds)
    try:
        # pylint: disable=exec-used
        exec(code, {}, {"poptorch": poptorch, "options": options})
    except SyntaxError as err:
        err_class = err.__class__.__name__
        detail = err.args[0]
        lineno = err.lineno
        line = err.text
        # pylint: disable=no-member
        raise poptorch.ConfigFileError("{} at line {} of {}: {}\n> {}".format(
            err_class, lineno, filename, detail, line))
