#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import ctypes
import functools
import inspect
import itertools
import json
from typing import List, Generator

import torch

from . import poptorch_core  # type: ignore
from ._logging import logger

ATTR_PREFIX = "attr:"


def deprecated(domain, since_version, reason):
    """Decorator function to mark other functions as deprecated."""

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


def accessAttributes(attribute_id_str):
    """Allow access to attributes"""
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
    return x.device.type == "ipu"


custom_arg_parsers = dict()


def getCustomParser(custom_type_instance):
    if len(custom_arg_parsers) == 0:
        return None

    # direct lookup for exact type inside custom_arg_parsers
    parser = custom_arg_parsers.get(type(custom_type_instance), None)
    if parser is not None:
        return parser

    # search for registered parser for base class of custom_type_instance,
    # iterate over entire dict
    for custom_type, parser in custom_arg_parsers.items():
        if isinstance(custom_type_instance, custom_type):
            return parser

    return None


# Returns the structure `tensors` as a list of its torch.Tensor contents.
def flattenTensorStructure(tensors, canonical_structure=None):
    def flatten(x, c):
        parser = getCustomParser(x)
        if parser is not None:
            yield from parser.yieldTensors(x)
        elif isinstance(x, dict):
            keys = x.keys() if c is None else c.keys()
            for k in keys:
                yield from flatten(x[k], None if c is None else c[k])
        elif isinstance(x, (list, tuple)):
            cl = itertools.repeat(None, len(x)) if c is None else c
            for t, ct in zip(x, cl):
                yield from flatten(t, ct)
        elif isinstance(x, torch.Tensor):
            yield x
        # If it's not a dict/list/tuple or tensor, just ignore it

    return list(flatten(tensors, canonical_structure))


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
        parser = getCustomParser(x)
        if parser is not None:
            return parser.reconstruct(x, it)
        if isinstance(x, dict):
            return type(x)({k: copy_structure(x[k], it) for k in x.keys()})
        if isinstance(x, (tuple, list)):
            return type(x)(copy_structure(e, it) for e in x)
        if isinstance(x, torch.Tensor) and filter_fn(x):
            return next(it)
        return x

    return copy_structure(structure, iter(values))


def combine_batch_tensors_gen(tensors: List[List[torch.Tensor]]
                              ) -> Generator[torch.Tensor, None, None]:
    """Concatenated batches tensors along dim = 0.
    """
    for tensor_id in range(len(tensors[0])):
        tensors_list = [
            tensors[batch_id][tensor_id] for batch_id in range(len(tensors))
        ]
        yield torch.cat(tensors_list)


def combined_batch_generator(dataloader_iterator,
                             num_batches_to_combine,
                             drop_last=True):
    """Wraps DataLoader iterator. Generates combined batches by concatenating
    consecutive batches tensors from dataloader_iterator along dim=0.
    """
    tensors_to_combine = []
    batch = None
    # iterate over next data batches
    for batch in dataloader_iterator:
        # append batch tensors to concatenate list
        if len(tensors_to_combine) < num_batches_to_combine:
            tensors_to_combine.append(flattenTensorStructure(batch))
        else:
            # concatenate all tensors from concatenate list - create combined batch
            yield reconstructTensorStructure(
                batch, combine_batch_tensors_gen(tensors_to_combine))
            tensors_to_combine = [flattenTensorStructure(batch)]

    if tensors_to_combine and len(tensors_to_combine) > 0 and \
        len(tensors_to_combine) == num_batches_to_combine or \
        not drop_last:
        # concatenate all tensors from concatenate list - create combined batch
        yield reconstructTensorStructure(
            batch, combine_batch_tensors_gen(tensors_to_combine))


def getIpuTensorId(x: torch.Tensor):
    return poptorch_core.getIpuTensorId(x)
