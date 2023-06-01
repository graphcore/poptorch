# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
# This file includes content from PyTorch Cluster which
# has been modified by Graphcore Ltd.
#
# Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch

from torch import Tensor
from torch_geometric.typing import OptTensor


def radius(
        x: Tensor,
        y: Tensor,
        r: float,
        batch_x: OptTensor = None,
        batch_y: OptTensor = None,
        max_num_neighbors: int = 32,
        *args,
        **kwargs,
) -> Tensor:
    # pylint: disable=unused-argument, keyword-arg-before-vararg
    r"""Computes graph edges to all points within a given distance.

    .. code-block:: python

        import torch
        from torch_geometric.nn import radius_graph

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)
        flow (str, optional): The flow direction when using in combination with
            message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`torch.Tensor` with static shape, where not found neighbours
            are marked by -1
    """
    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    if not torch.is_floating_point(x):
        x = x.float()

    if not torch.is_floating_point(y):
        y = y.float()

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    x = torch.cat([x, 2 * r * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * r * batch_y.view(-1, 1).to(y.dtype)], dim=-1)
    distance_upper_bound = r + 1e-8

    dist = torch.cdist(y, x)
    k = min(dist.size(-1), max_num_neighbors)
    dist, col = torch.topk(dist, k=k, dim=-1, largest=False, sorted=True)
    row = torch.cat([torch.full(col.shape[1:], i) for i in range(col.size(0))],
                    dim=0)
    col = torch.where(dist < distance_upper_bound, col, -1)
    col = torch.flatten(col)
    row = torch.where(col == -1, -1, row)

    return torch.stack([row, col], dim=0)


def radius_graph(
        x: torch.Tensor,
        r: float,
        batch: OptTensor = None,
        loop: bool = False,
        max_num_neighbors: int = 32,
        flow: str = 'source_to_target',
        *args,
        **kwargs,
) -> torch.Tensor:
    # pylint: disable=unused-argument, keyword-arg-before-vararg
    r"""Computes graph edges to all points within a given distance.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch` needs to be sorted.
            (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import radius_graph

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)
    """

    assert flow in ['source_to_target', 'target_to_source']
    edge_index = radius(x, x, r, batch, batch,
                        max_num_neighbors if loop else max_num_neighbors + 1,
                        *args, **kwargs)
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)
