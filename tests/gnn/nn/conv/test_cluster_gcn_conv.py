# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
from torch_geometric.nn import ClusterGCNConv
from conv_utils import conv_harness


def test_cluster_gcn_conv(dataset, request):
    pytest.skip(
        f'{request.node.nodeid}: AFS-145: Operations using aten::nonzero '
        'are unsupported because the output shape is determined by the '
        'tensor values. The IPU cannot support dynamic output shapes')

    in_channels = dataset.num_node_features
    out_channels = 32
    conv = ClusterGCNConv(in_channels,
                          out_channels,
                          diag_lambda=1.,
                          add_self_loops=False)
    conv_harness(conv, dataset)
