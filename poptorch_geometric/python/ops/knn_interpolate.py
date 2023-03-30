# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch

from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter

from poptorch_geometric.ops.knn import knn


def knn_interpolate(x: torch.Tensor,
                    pos_x: torch.Tensor,
                    pos_y: torch.Tensor,
                    batch_x: OptTensor = None,
                    batch_y: OptTensor = None,
                    k: int = 3,
                    num_workers: int = 1):
    r"""The k-NN interpolation from the `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper.
    For each point :math:`y` with position :math:`\mathbf{p}(y)`, its
    interpolated features :math:`\mathbf{f}(y)` are given by
    .. math::
        \mathbf{f}(y) = \frac{\sum_{i=1}^k w(x_i) \mathbf{f}(x_i)}{\sum_{i=1}^k
        w(x_i)} \textrm{, where } w(x_i) = \frac{1}{d(\mathbf{p}(y),
        \mathbf{p}(x_i))^2}
    and :math:`\{ x_1, \ldots, x_k \}` denoting the :math:`k` nearest points
    to :math:`y`.
    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        pos_x (torch.Tensor): Node position matrix
            :math:`\in \mathbb{R}^{N \times d}`.
        pos_y (torch.Tensor): Upsampled node position matrix
            :math:`\in \mathbb{R}^{M \times d}`.
        batch_x (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b_x} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node from :math:`\mathbf{X}` to a specific example.
            (default: :obj:`None`)
        batch_y (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b_y} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node from :math:`\mathbf{Y}` to a specific example.
            (default: :obj:`None`)
        k (int, optional): Number of neighbors. (default: :obj:`3`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
    """

    with torch.no_grad():
        assign_index = knn(pos_x,
                           pos_y,
                           k,
                           batch_x=batch_x,
                           batch_y=batch_y,
                           num_workers=num_workers)
        y_idx, x_idx = assign_index[0], assign_index[1]

        extended_y_idx = torch.where(y_idx == -1, pos_y.size(0), y_idx)
        extended_x_idx = torch.where(x_idx == -1, pos_x.size(0), x_idx)
        posx_zeros = torch.zeros_like(pos_x[:1])
        extended_diff_x = torch.cat((pos_x, posx_zeros))
        posy_zeros = torch.zeros_like(pos_y[:1])
        extended_diff_y = torch.cat((pos_y, posy_zeros))

        x_zeros = torch.zeros_like(x[:1])
        extended_x = torch.cat((x, x_zeros))

        diff = torch.index_select(extended_diff_x, 0,
                                  extended_x_idx) - torch.index_select(
                                      extended_diff_y, 0, extended_y_idx)
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

    scatter_y_idx = torch.arange(pos_y.size(0),
                                 dtype=torch.long).view(-1,
                                                        1).repeat(1,
                                                                  k).view(-1)
    y = scatter(torch.index_select(extended_x, 0, extended_x_idx) * weights,
                scatter_y_idx,
                0,
                pos_y.size(0),
                reduce='sum')
    return y / scatter(weights, scatter_y_idx, 0, pos_y.size(0), reduce='sum')
