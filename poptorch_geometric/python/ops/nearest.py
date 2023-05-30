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

from typing import List, Optional, Union

import torch


def _batch_size(batch_x, batch_y):
    batch_x_size = 0
    if isinstance(batch_x, list):
        batch_x_size = len(batch_x)
        if batch_x is not None and any(
            (batch_x[i] < batch_x[i - 1] for i in range(1, batch_x_size))):
            raise ValueError("'batch_x' is not sorted")
    elif isinstance(batch_x, torch.Tensor):
        batch_x_size = batch_x.size()[0]

    batch_y_size = 0
    if isinstance(batch_y, list):
        batch_y_size = len(batch_y)
        if batch_y is not None and any(
            (batch_y[i] < batch_y[i - 1] for i in range(1, batch_y_size))):
            raise ValueError("'batch_y' is not sorted")
    elif isinstance(batch_y, torch.Tensor):
        batch_y_size = batch_y.size()[0]

    return batch_x_size, batch_y_size


def _validate_batch_sorted(batch_x, batch_y):
    if isinstance(batch_x, list) and isinstance(batch_y, list):
        unique_batch_x = [batch_x[0]] + [
            batch_x[i]
            for i in range(1, len(batch_x)) if batch_x[i - 1] != batch_x[i]
        ]
        unique_batch_y = [batch_y[0]] + [
            batch_y[i]
            for i in range(1, len(batch_y)) if batch_y[i - 1] != batch_y[i]
        ]
        if unique_batch_x != unique_batch_y:
            raise ValueError("Some batch indices occur in 'batch_x' "
                             "that do not occur in 'batch_y'")


def _batch_dim_view(batch_x, batch_y):
    if isinstance(batch_x, torch.Tensor):
        batch_x_dim = batch_x.dim()
        batch_x_view = batch_x.view(-1, 1)
    else:
        batch_x_dim = 1
        batch_x_view = torch.tensor(batch_x, dtype=torch.long).view(-1, 1)

    if isinstance(batch_y, torch.Tensor):
        batch_y_dim = batch_y.dim()
        batch_y_view = batch_y.view(-1, 1)
    else:
        batch_y_dim = 1
        batch_y_view = torch.tensor(batch_y, dtype=torch.long).view(-1, 1)

    return batch_x_dim, batch_y_dim, batch_x_view, batch_y_view


def nearest(
        x: torch.Tensor,
        y: torch.Tensor,
        batch_x: Optional[Union[torch.Tensor, List[int]]] = None,
        batch_y: Optional[Union[torch.Tensor, List[int]]] = None,
) -> torch.Tensor:
    r"""Clusters points in :obj:`x` together which are nearest to a given query
    point in :obj:`y`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        batch_x (LongTensor or List[int], optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        batch_y (LongTensor or List[int], optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. :obj:`batch_y` needs to be sorted.
            (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import nearest

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])
        cluster = nearest(x, y, batch_x, batch_y)
    """

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    assert x.size(1) == y.size(1)

    batch_x_size, batch_y_size = _batch_size(batch_x, batch_y)

    if batch_x is None and batch_y is not None:
        batch_x = [0] * x.size(0)
        batch_x_size = x.size(0)
    if batch_y is None and batch_x is not None:
        batch_y = [0] * y.size(0)
        batch_y_size = y.size(0)

    # Translate and rescale x and y to [0, 1].
    if batch_x is not None and batch_y is not None:
        # If an instance in `batch_x` is non-empty, it must be non-empty in
        # `batch_y `as well:
        _validate_batch_sorted(batch_x, batch_y)

        batch_x_dim, batch_y_dim, batch_x_view, batch_y_view = _batch_dim_view(
            batch_x, batch_y)

        assert x.dim() == 2 and batch_x_dim == 1
        assert y.dim() == 2 and batch_y_dim == 1
        assert x.size(0) == batch_x_size
        assert y.size(0) == batch_y_size

        min_xy = torch.min(x.min(), y.min())
        x, y = x - min_xy, y - min_xy

        max_xy = torch.max(x.max(), y.max())
        x = torch.div(x, max_xy)
        y = torch.div(y, max_xy)

        # Concat batch/features to ensure no cross-links between examples.
        D = x.size(-1)
        x = torch.cat([x, 2. * D * batch_x_view], -1)
        y = torch.cat([y, 2. * D * batch_y_view], -1)

    distances = torch.cdist(x.float(), y.float())
    indices = torch.argmin(distances, dim=1)
    return indices
