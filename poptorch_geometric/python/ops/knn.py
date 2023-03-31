# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional

import torch


def knn(x: torch.Tensor,
        y: torch.Tensor,
        k: int,
        batch_x: Optional[torch.Tensor] = None,
        batch_y: Optional[torch.Tensor] = None,
        *args,
        **kwargs):
    # pylint: disable=unused-argument, keyword-arg-before-vararg
    r"""Finds for each element in `y` the `k` nearest points in `x`.

    Args:
        x (torch.Tensor): Node feature matrix
        y (torch.Tensor): Node feature matrix
        k (int): The number of neighbors.
        batch_x (torch.Tensor, optional): Batch vector which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (torch.Tensor, optional): Batch vector which assigns each
            node to a specific example. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_cluster import knn

    .. testcode::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch_x = torch.tensor([0, 0, 0, 0])
        >>> y = torch.Tensor([[-1, 0], [1, 0]])
        >>> batch_y = torch.tensor([0, 0])
        >>> assign_index = knn(x, y, 2, batch_x, batch_y)
    """

    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.int32)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.int32)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    # Rescale x and y.
    min_xy = torch.min(torch.min(x), torch.min(y))
    x, y = x - min_xy, y - min_xy

    max_xy = torch.max(torch.max(x), torch.max(y))
    x, y, = x / max_xy, y / max_xy

    # Concat batch/features to ensure no cross-links between examples exist.
    x = torch.cat([
        x, 2 * x.size(1) * batch_x.view(
            -1, 1).to(torch.int32 if x.dtype == torch.long else x.dtype)
    ],
                  dim=-1)
    y = torch.cat([
        y, 2 * y.size(1) * batch_y.view(
            -1, 1).to(torch.int32 if y.dtype == torch.long else y.dtype)
    ],
                  dim=-1)

    x_expanded = x.expand(y.size(0), *x.shape)
    y_expanded = y.reshape(y.size(0), 1, y.size(1))

    dist, col = torch.topk(torch.norm(x_expanded - y_expanded, dim=-1),
                           k=k,
                           dim=-1,
                           largest=False,
                           sorted=True)
    row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)

    distance_upper_bound = x.size(1)

    row = torch.where(dist > distance_upper_bound, -1, row).view(-1)
    col = torch.where(dist > distance_upper_bound, -1, col).view(-1)

    return torch.stack([row, col], dim=0)
