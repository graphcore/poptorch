#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import functools

from ._logging import logger


# Decorator function to mark other functions as
# deprecated.
def deprecated(domain, since_version, reason):
    def deprecated_func(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            logger.warning(
                "%s.%s is deprecated since version %s "
                "and will be removed in a future release.\nReason: %s.",
                domain, func.__name__, since_version, reason)
            return func(*args, **kwargs)

        return wrapped_func

    return deprecated_func
