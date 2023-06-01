# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch
import torch_cluster

from poptorch_geometric.ops.radius import radius, radius_graph
import poptorch


def to_set(edge_index):
    # pylint: disable=R1721
    return {(i, j) for i, j in edge_index.t().tolist()}


def assert_fn(native_out, poptorch_out):
    poptorch_out = poptorch_out[poptorch_out != -1]
    dim = poptorch_out.size(0) // 2
    poptorch_out = poptorch_out.reshape((2, dim))

    native_out = native_out[native_out != -1]
    dim = native_out.size(0) // 2
    native_out = native_out.reshape((2, dim))

    assert to_set(poptorch_out) == to_set(native_out)


def op_harness(op, reference_op, *args, **kwargs):

    native_out = reference_op(*args, **kwargs)

    class Model(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return op(*args, **kwargs)

    model = poptorch.inferenceModel(Model())

    poptorch_out = model(*args, **kwargs)

    assert_fn(native_out, poptorch_out)


@pytest.mark.parametrize("with_batch", [True, False])
def test_radius_basic(with_batch):
    x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    y = torch.Tensor([[-1, 0], [1, 0]])

    if with_batch:
        batch_x = torch.tensor([0, 0, 0, 1])
        batch_y = torch.tensor([0, 1])
    else:
        batch_x = None
        batch_y = None

    op_harness(radius, torch_cluster.radius, x, y, 1.5, batch_x, batch_y)


def test_radius_upstream():
    x = torch.tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -10],
    ])
    y = torch.tensor([
        [0, 0],
        [0, 1],
    ])

    batch_x = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    batch_y = torch.tensor([0, 1], dtype=torch.long)

    op_harness(radius, torch_cluster.radius, x, y, 2, max_num_neighbors=4)
    op_harness(radius,
               torch_cluster.radius,
               x,
               y,
               2,
               batch_x,
               batch_y,
               max_num_neighbors=4)

    # Skipping a batch
    batch_x = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2], dtype=torch.long)
    batch_y = torch.tensor([0, 2], dtype=torch.long)
    op_harness(radius,
               torch_cluster.radius,
               x,
               y,
               2,
               batch_x,
               batch_y,
               max_num_neighbors=4)


@pytest.mark.parametrize('flow', ['source_to_target', 'target_to_source'])
def test_radius_graph(flow):
    x = torch.tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ])

    op_harness(radius_graph,
               torch_cluster.radius_graph,
               x,
               r=2.5,
               loop=True,
               flow=flow)


@pytest.mark.ipuHardwareRequired
def test_radius_graph_large():
    torch.manual_seed(40)
    x = torch.randn(1000, 3)

    op_harness(radius_graph,
               torch_cluster.radius_graph,
               x,
               r=2.5,
               loop=True,
               flow='target_to_source',
               max_num_neighbors=2000)
