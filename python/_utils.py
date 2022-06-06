#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import ctypes
import functools
import inspect
import json

import torch

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


def on_ipu(x):
    # TODO(T59880) rename xla -> ipu
    return x.device.type == "xla"


def flattenTensorStructure(tensors):
    def flatten(x):
        if isinstance(x, dict):
            for t in x.values():
                yield from flatten(t)
        elif isinstance(x, (list, tuple)):
            for t in x:
                yield from flatten(t)
        elif isinstance(x, torch.Tensor):
            yield x
        # If it's not a dict/list/tuple or tensor, just ignore it

    return list(flatten(tensors))


# Turns a flat 'output' into the same structure as 'outputs_structure'.
def reconstructTensorStructure(outputs_structure, output):
    # Copy the original structure but replace all the tensors
    # by values from the passed iterator.
    def copy_structure(x, it):
        if isinstance(x, dict):
            return {k: copy_structure(v, it) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return type(x)(copy_structure(e, it) for e in x)
        if isinstance(x, torch.Tensor):
            return next(it)
        return x

    return copy_structure(outputs_structure, iter(output))


# Replace the ipu tensors in the output structure with values from output
def replaceIpuTensors(outputs_structure, output):
    # Copy the original structure but replace all the tensors
    # by values from the passed iterator.
    def copy_structure(x, it):
        if isinstance(x, dict):
            return {k: copy_structure(v, it) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return type(x)(copy_structure(e, it) for e in x)
        if isinstance(x, torch.Tensor) and on_ipu(x):
            return next(it)
        return x

    return copy_structure(outputs_structure, iter(output))
