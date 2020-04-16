import torch
import torch.nn as nn
import numpy as np

import poptorch


# Return true if `graph` contains a node of `kind`.
def _has_node(graph, kind):
    for node in graph.nodes():
        if node.kind() == kind:
            return True
    return False


def _getOutputShape(graph):
    return list(graph.outputs())[0].type().sizes()


def test_conv2d():
    class X(nn.Module):
        def __init__(self):
            super(X, self).__init__()
            self.conv2d = nn.Conv2d(5, 3, 3)

        def forward(self, x):
            return self.conv2d(x)

    # Create a module, run it, and get the shape of the returned output.
    m = X()
    dummyInput = torch.zeros(1, 5, 30, 30)
    actualOutputShape = list(m(dummyInput).size())

    # jit.script the module and run some passes to shrink the graph.
    # Shrinking the graph is mostly done for debugability.
    m = torch.jit.script(m)
    graph = m.graph
    torch._C._jit_pass_inline(graph)
    torch._C._jit_pass_constant_propagation(graph)
    graph, params = torch._C._jit_pass_lower_graph(graph, m._c)
    # Observe the graph doesn't already have a shape for the output.
    assert _getOutputShape(graph) is None

    # Run shape analysis on the graph
    poptorch.propagateInputShapes(graph, (dummyInput, ))
    inferedOutputShape = _getOutputShape(graph)

    assert _has_node(graph, 'aten::conv2d')
    assert inferedOutputShape == actualOutputShape


def test_batchnorm():
    class X(nn.Module):
        def __init__(self, features):
            super(X, self).__init__()
            self.bnorm = nn.BatchNorm1d(features)

        def forward(self, x):
            return self.bnorm(x)

    # Create a module, run it, and get the shape of the returned output.
    m = X(100)
    dummyInput = torch.zeros(20, 100)
    actualOutputShape = list(m(dummyInput).size())

    # jit.script the module and run some passes to shrink the graph.
    # Shrinking the graph is mostly done for debugability.
    m = torch.jit.script(m)
    graph = m.graph
    torch._C._jit_pass_inline(graph)
    poptorch.poptorch_core.peepholeOptimizations(graph, False);
    graph, params = torch._C._jit_pass_lower_graph(graph, m._c)
    torch._C._jit_pass_constant_propagation(graph)
    # Observe the graph doesn't already have a shape for the output.
    assert _getOutputShape(graph) is None

    # Run shape analysis on the graph
    poptorch.propagateInputShapes(graph, (dummyInput, ))
    inferedOutputShape = _getOutputShape(graph)

    assert _has_node(graph, 'aten::batch_norm')
    assert inferedOutputShape == actualOutputShape


def test_maxpool2d():
    class X(nn.Module):
        def __init__(self, *args, **kwargs):
            super(X, self).__init__()
            self.pool = nn.MaxPool2d(*args, **kwargs)

        def forward(self, x):
            return self.pool(x)

    # Create a module, run it, and get the shape of the returned output.
    m = X(3, stride=2)
    dummyInput = torch.zeros(20, 16, 50, 32)
    actualOutputShape = list(m(dummyInput).size())

    # jit.script the module and run some passes to shrink the graph.
    # Shrinking the graph is mostly done for debugability.
    m = torch.jit.script(m)
    graph = m.graph
    torch._C._jit_pass_inline(graph)
    torch._C._jit_pass_constant_propagation(graph)
    graph, params = torch._C._jit_pass_lower_graph(graph, m._c)
    # Observe the graph doesn't already have a shape for the output.
    assert _getOutputShape(graph) is None

    # Run shape analysis on the graph
    poptorch.propagateInputShapes(graph, (dummyInput, ))
    inferedOutputShape = _getOutputShape(graph)

    assert _has_node(graph, 'aten::max_pool2d')
    assert inferedOutputShape == actualOutputShape

def test_view():
    class X(nn.Module):
        def __init__(self, *args, **kwargs):
            super(X, self).__init__()

        def forward(self, x):
            return x.view(50, -1)

    m = X()
    dummyInput = torch.zeros(100, 100)
    actualOutputShape = list(m(dummyInput).size())

    print(actualOutputShape)
    m = torch.jit.script(m)
    graph = m.graph
    graph, params = torch._C._jit_pass_lower_graph(graph, m._c)
    torch._C._jit_pass_constant_propagation(graph)
    assert _getOutputShape(graph) is None

    poptorch.propagateInputShapes(graph, (dummyInput, ))
    inferedOutputShape = _getOutputShape(graph)

    assert _has_node(graph, 'aten::view')
    assert inferedOutputShape == actualOutputShape

# "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor") ||
def test_addmm():
    class X(nn.Module):
        def __init__(self, *args, **kwargs):
            super(X, self).__init__()

        def forward(self, x, y, z):
            return torch.addmm(x, y, z)

    m =  X()
    dummyInputs = (torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(3, 4))
    actualOutputShape = list(m(*dummyInputs).size())

    m = torch.jit.script(m)
    graph = m.graph
    graph, params = torch._C._jit_pass_lower_graph(graph, m._c)
    assert _getOutputShape(graph) is None

    poptorch.propagateInputShapes(graph, dummyInputs)
    inferedOutputShape = _getOutputShape(graph)

    assert _has_node(graph, 'aten::addmm')
    assert inferedOutputShape == actualOutputShape


