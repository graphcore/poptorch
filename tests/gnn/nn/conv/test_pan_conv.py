# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
from torch_geometric.nn import PANConv

from conv_utils import conv_harness


@pytest.mark.skip(reason='AFS-144: Error: NotImplementedError: Could not run '
                  'aten::_local_scalar_dense')
def test_pan_conv(dataset):
    in_channels = dataset.num_node_features
    conv = PANConv(in_channels, 32, filter_size=2, add_self_loops=False)

    conv_harness(conv, dataset)
