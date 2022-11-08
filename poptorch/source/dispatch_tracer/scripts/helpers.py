# Copyright (c) 2022 Graphcore Ltd. All rights reserved.


def addScope(string, scope='    '):
    return '\n'.join(s if s == '' else scope + s for s in string.split('\n'))
