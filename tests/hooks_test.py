# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import torch
import torch.nn as nn

from poptorch import (inferenceModel, registerPostCompileHook,
                      registerPreCompileHook)


class Model(nn.Module):
    def forward(self, input):
        return input


def test_precompile_and_postcompile_hooks():
    """Test that registered pre and post compile hooks are called."""
    model = Model()

    precompile_called = False
    postcompile_called = False

    def precompile():
        nonlocal precompile_called
        precompile_called = True

    def postcompile():
        nonlocal postcompile_called
        postcompile_called = True

    registerPreCompileHook(precompile)
    registerPostCompileHook(postcompile)

    poplar_exec = inferenceModel(model)
    input = torch.randn((10, 10), dtype=torch.float32)
    poplar_exec(input)
    assert precompile_called and postcompile_called


def test_non_callable():
    """Test that an error is raised if a non-callable
    is attempted to be registered"""
    with pytest.raises(RuntimeError, match="must be callable"):
        registerPreCompileHook(2)

    with pytest.raises(RuntimeError, match="must be callable"):
        registerPostCompileHook(False)


def test_called_in_order():
    """Test that hooks are called in the order they were registered in."""
    expected_calls = [1, 2, 3]
    calls = []

    def hookO():
        nonlocal calls
        calls.append(expected_calls[0])

    def hook1():
        nonlocal calls
        calls.append(expected_calls[1])

    def hook2():
        nonlocal calls
        calls.append(expected_calls[2])

    registerPreCompileHook(hookO)
    registerPreCompileHook(hook1)
    registerPreCompileHook(hook2)

    model = Model()
    poplar_exec = inferenceModel(model)
    input = torch.randn((10, 10), dtype=torch.float32)
    poplar_exec(input)

    assert calls == expected_calls


def test_can_remove():
    """Test that a hook is correctly removed via Torch's RemovableHandle."""
    called = False

    def hook():
        nonlocal called
        called = True

    handle = registerPostCompileHook(hook)
    handle.remove()

    model = Model()
    poplar_exec = inferenceModel(model)
    input = torch.randn((10, 10), dtype=torch.float32)
    poplar_exec(input)

    assert not called
