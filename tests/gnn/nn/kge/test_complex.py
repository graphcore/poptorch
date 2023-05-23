# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch

from torch_geometric.nn import ComplEx
from kge_utils import kge_harness


def test_complex_scoring():
    model = ComplEx(num_nodes=5, num_relations=2, hidden_channels=1)

    model.node_emb.weight.data = torch.tensor([
        [2.],
        [3.],
        [5.],
        [1.],
        [2.],
    ])
    model.node_emb_im.weight.data = torch.tensor([
        [4.],
        [1.],
        [3.],
        [1.],
        [2.],
    ])
    model.rel_emb.weight.data = torch.tensor([
        [2.],
        [3.],
    ])
    model.rel_emb_im.weight.data = torch.tensor([
        [3.],
        [1.],
    ])

    head_index = torch.tensor([1, 3])
    rel_type = torch.tensor([1, 0])
    tail_index = torch.tensor([2, 4])

    loader = model.loader(head_index, rel_type, tail_index, batch_size=5)
    kge_harness(model, loader)


def test_complex():
    model = ComplEx(num_nodes=10, num_relations=5, hidden_channels=32)
    assert str(model) == 'ComplEx(10, num_relations=5, hidden_channels=32)'

    head_index = torch.tensor([0, 2, 4, 6, 8])
    rel_type = torch.tensor([0, 1, 2, 3, 4])
    tail_index = torch.tensor([1, 3, 5, 7, 9])

    loader = model.loader(head_index, rel_type, tail_index, batch_size=5)
    kge_harness(model, loader)
