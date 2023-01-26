# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
from torch_geometric.nn import FiLMConv
from conv_utils import conv_harness

out_channels = 32


@pytest.mark.parametrize('num_relations', [1])
def test_film_conv(dataset, num_relations):
    in_channels = dataset.num_node_features
    conv = FiLMConv(in_channels, out_channels, num_relations=num_relations)

    conv_harness(conv, dataset)
