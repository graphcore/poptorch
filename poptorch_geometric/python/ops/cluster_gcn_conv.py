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

# Modified version of ClusterGCNConv that does not use dynamic shapes.
# Original pytorch_geometric version will be replaced by this code.
import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, SparseTensor, torch_sparse
from torch_geometric.utils import (
    add_self_loops,
    degree,
    remove_self_loops,
    spmm,
)


# pylint: disable=abstract-method, arguments-differ, no-value-for-parameter
class ClusterGCNConv(MessagePassing):
    r"""The ClusterGCN graph convolutional operator from the
    `"Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph
    Convolutional Networks" <https://arxiv.org/abs/1905.07953>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \left( \mathbf{\hat{A}} + \lambda \cdot
        \textrm{diag}(\mathbf{\hat{A}}) \right) \mathbf{X} \mathbf{W}_1 +
        \mathbf{X} \mathbf{W}_2
    where :math:`\mathbf{\hat{A}} = {(\mathbf{D} + \mathbf{I})}^{-1}(\mathbf{A}
    + \mathbf{I})`.
    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        diag_lambda (float, optional): Diagonal enhancement value
            :math:`\lambda`. (default: :obj:`0.`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 diag_lambda: float = 0.,
                 add_self_loops: bool = True,
                 bias: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.diag_lambda = diag_lambda
        self.add_self_loops = add_self_loops

        self.lin_out = Linear(in_channels,
                              out_channels,
                              bias=bias,
                              weight_initializer='glorot')
        self.lin_root = Linear(in_channels,
                               out_channels,
                               bias=False,
                               weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_out.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        edge_weight: OptTensor = None
        if isinstance(edge_index, Tensor):
            num_nodes = x.size(self.node_dim)
            if self.add_self_loops:
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

            row, col = edge_index[0], edge_index[1]
            deg_inv = 1. / degree(col, num_nodes=num_nodes).clamp_(1.)

            edge_weight = deg_inv[col]
            eq = torch.eq(row, col)
            broadcast = torch.index_select(self.diag_lambda * deg_inv, 0, row)
            tmp = torch.mul(eq.float(), broadcast)
            edge_weight = torch.add(edge_weight, tmp)

        elif isinstance(edge_index, SparseTensor):
            if self.add_self_loops:
                edge_index = torch_sparse.set_diag(edge_index)

            col, row, _ = edge_index.coo()  # Transposed.
            deg_inv = 1. / torch_sparse.sum(edge_index, dim=1).clamp_(1.)

            edge_weight = deg_inv[col]
            eq = torch.eq(row, col)
            broadcast = torch.index_select(self.diag_lambda * deg_inv, 0, row)
            tmp = torch.mul(eq.float(), broadcast)
            edge_weight = torch.add(edge_weight, tmp)
            edge_index = edge_index.set_value(edge_weight, layout='coo')

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index,
                             x=x,
                             edge_weight=edge_weight,
                             size=None)
        out = self.lin_out(out) + self.lin_root(x)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, diag_lambda={self.diag_lambda})')
