# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
# This file includes content from PyTorch Geometric which
# has been modified by Graphcore Ltd.
#
# Copyright (c) 2023 PyG Team <team@pyg.org>
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

from typing import Union, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.utils import scatter


def to_dense_batch(x: Tensor,
                   batch: Optional[Tensor] = None,
                   fill_value: Union[Tensor, float] = 0.,
                   max_num_nodes: Optional[int] = None,
                   batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    r"""Given a sparse batch of node features
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}` (with
    :math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
    dense node feature tensor
    :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
    :math:`N_{\max} = \max_i^B N_i`).
    In addition, a mask of shape :math:`\mathbf{M} \in \{ 0, 1 \}^{B \times
    N_{\max}}` is returned, holding information about the existence of
    fake-nodes in the dense representation.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered. (default: :obj:`None`)
        fill_value (float, optional): The value for invalid entries in the
            resulting dense output tensor. (default: :obj:`0`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)
        batch_size (int, optional) The batch size. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`BoolTensor`)
    """

    if batch is None and max_num_nodes is None:
        mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
        return x.unsqueeze(0), mask

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_size is None:
        raise RuntimeError(
            "IPU MK2 doesn't support dynamic shapes. `batch_size` argument "
            "must be specified.")

    if max_num_nodes is None:
        raise RuntimeError(
            "IPU MK2 doesn't support dynamic shapes. `max_num_nodes` argument "
            "must be specified.")

    num_nodes = scatter(batch.new_ones(x.size(0)),
                        batch,
                        dim=0,
                        dim_size=batch_size,
                        reduce='sum')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    tmp = torch.arange(batch.size(0), device=x.device) - cum_nodes[batch]
    idx = tmp + (batch * max_num_nodes)
    x_size_slice = list(x.size())[1:]

    size = [batch_size * max_num_nodes] + x_size_slice
    if isinstance(fill_value, torch.Tensor):
        out = fill_value.repeat(*size)
    else:
        out = x.new_full(size, fill_value)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + x_size_slice)

    mask = torch.zeros(batch_size * max_num_nodes,
                       dtype=torch.bool,
                       device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask
