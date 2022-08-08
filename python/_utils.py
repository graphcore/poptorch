#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import ctypes
import functools
import inspect
import json

import torch

from . import poptorch_core  # type: ignore
from ._logging import logger
from .ops import ATTR_PREFIX


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


def assert_signatures_match(poptorch_method, reference_method):
    reference_params = inspect.signature(reference_method).parameters
    poptorch_params = inspect.signature(poptorch_method).parameters
    assert poptorch_params == reference_params, (
        "Arguments mismatch: expected "
        f"{reference_params} but got {poptorch_params}")


# Allow access to attributes
def accessAttributes(attribute_id_str):
    logger.debug("Accessing attributes with: %s", attribute_id_str)

    if not isinstance(attribute_id_str, (str)):
        raise ValueError("Wrong type for attribute_id_str")

    # this is to allow creating of attributes from poptorch cpp
    if attribute_id_str.startswith('{'):
        return json.loads(attribute_id_str)

    if not attribute_id_str.startswith(ATTR_PREFIX):
        raise ValueError("Invalid attribute_id_str")

    attribute_id = int(attribute_id_str[len(ATTR_PREFIX):], 16)

    # NB this is undefined behavior if attribute_id does not exist
    attributes = ctypes.cast(attribute_id, ctypes.py_object).value
    logger.debug(str(attributes))

    if attributes is None:
        return {}
    return attributes


def isOnIpu(x):
    # TODO(T59880) rename xla -> ipu
    return x.device.type == "xla"


custom_arg_parsers = dict()


# Returns the structure `tensors` as a list of its torch.Tensor contents.
def flattenTensorStructure(tensors):
    def flatten(x):
        if isinstance(x, dict):
            for k in sorted(x.keys()):
                yield from flatten(x[k])
        elif isinstance(x, (list, tuple)):
            for t in x:
                yield from flatten(t)
        elif isinstance(x, torch.Tensor):
            yield x
        for custom_type, parser in custom_arg_parsers.items():
            if isinstance(x, custom_type):
                yield from parser.yieldTensors(x)
        # If it's not a dict/list/tuple or tensor, just ignore it

    return list(flatten(tensors))


# Turns a flat `values` into the same structure as `structure`.
#
# Any non-tensor values in `structure` will be copied to the output.
#
# filter_fn: Optional function to additionally filter which tensors make it into
#            the output (eg. could supply `isOnIpu` to only get IPU tensors).
def reconstructTensorStructure(structure, values, filter_fn=lambda t: True):
    # Copy the original structure but replace all the tensors by values from the
    # passed iterator.
    def copy_structure(x, it):
        if isinstance(x, dict):
            return {k: copy_structure(x[k], it) for k in sorted(x.keys())}
        if isinstance(x, (tuple, list)):
            return type(x)(copy_structure(e, it) for e in x)
        if isinstance(x, torch.Tensor) and filter_fn(x):
            return next(it)
        for custom_type, parser in custom_arg_parsers.items():
            if isinstance(x, custom_type):
                return parser.reconstruct(x, it)
        return x

    return copy_structure(structure, iter(values))


def getIpuTensorId(x: torch.Tensor):
    return poptorch_core.getIpuTensorId(x)
