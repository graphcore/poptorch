# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pytest
import torch
from torch_geometric import seed_everything
from torch_geometric.nn import RGATConv

from conv_utils import conv_harness


@pytest.mark.parametrize('mod', [
    'additive',
    'scaled',
    'f-additive',
    'f-scaled',
])
@pytest.mark.parametrize('attention_mechanism', [
    'within-relation',
    'across-relation',
])
@pytest.mark.parametrize('attention_mode', [
    'additive-self-attention',
    'multiplicative-self-attention',
])
def test_rgat_conv(mod, attention_mechanism, attention_mode, request):
    seed_everything(0)

    if attention_mechanism == 'within-relation':
        pytest.skip(
            f'{request.node.nodeid}: AFS-145: Operations using aten::nonzero '
            'are unsupported because the output shape is determined by the '
            'tensor values. The IPU cannot support dynamic output shapes.')

    elif mod != 'additive' or attention_mode != 'additive-self-attention':
        pytest.skip(
            f'{request.node.nodeid}: AFS-200: Various inconsistent failures '
            'or crashes for many confgurations.')

    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_type = torch.tensor([0, 2, 1, 2])
    edge_attr = torch.randn((4, 8))

    conv = RGATConv(8,
                    20,
                    num_relations=4,
                    num_bases=4,
                    mod=mod,
                    attention_mechanism=attention_mechanism,
                    attention_mode=attention_mode,
                    heads=2,
                    dim=1,
                    edge_dim=8,
                    add_self_loops=False)

    batch = (x, edge_index, edge_type, edge_attr)
    conv_harness(conv, batch=batch)
