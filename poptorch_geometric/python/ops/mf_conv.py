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

from typing import Union

import torch
import torch_geometric
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import degree
from torch import Tensor


# pylint: disable=abstract-method
class MFConv(torch_geometric.nn.conv.MFConv):
    r"""The graph neural network operator from the
    `"Convolutional Networks on Graphs for Learning Molecular Fingerprints"
    <https://arxiv.org/abs/1509.09292>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}^{(\deg(i))}_1 \mathbf{x}_i +
        \mathbf{W}^{(\deg(i))}_2 \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j

    which trains a distinct weight matrix for each possible vertex degree.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        max_degree (int, optional): The maximum node degree to consider when
            updating weights (default: :obj:`10`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        x_r = x[1]

        deg = x[0]  # Dummy.
        if isinstance(edge_index, SparseTensor):
            deg = edge_index.storage.rowcount()
        elif isinstance(edge_index, Tensor):
            i = 1 if self.flow == 'source_to_target' else 0
            N = x[0].size(self.node_dim)
            N = size[1] if size is not None else N
            N = x_r.size(self.node_dim) if x_r is not None else N
            deg = degree(edge_index[i], N, dtype=torch.long)
        deg.clamp_(max=self.max_degree)

        # propagate_type: (x: OptPairTensor)
        h = self.propagate(edge_index, x=x, size=size)

        out = h.new_empty(list(h.size())[:-1] + [self.out_channels])
        for i, (lin_l, lin_r) in enumerate(zip(self.lins_l, self.lins_r)):
            mask = torch.eq(deg, i).view(-1, 1)
            h_deg_i = torch.where(mask, h, 0.0)
            r = lin_l(h_deg_i)

            if x_r is not None:
                x_r_deg_i = torch.where(mask, x_r, 0.0)
                r = r + lin_r(x_r_deg_i)

            out = torch.where(mask, r, out)

        return out
