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


def unrollTensorList(tensor_or_list, accumulated_tensors=None):
    if accumulated_tensors is None:
        accumulated_tensors = []

    if isinstance(tensor_or_list, (list, tuple)):
        for t in tensor_or_list:
            accumulated_tensors = unrollTensorList(t, accumulated_tensors)
    elif isinstance(tensor_or_list, torch.Tensor):
        accumulated_tensors.append(tensor_or_list)
    # If it's not a list/tuple or tensor, just ignore it

    return accumulated_tensors
