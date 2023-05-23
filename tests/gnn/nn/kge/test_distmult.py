# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch

from torch_geometric.nn import DistMult
from kge_utils import kge_harness


def test_distmult():
    model = DistMult(num_nodes=10, num_relations=5, hidden_channels=32)
    assert str(model) == 'DistMult(10, num_relations=5, hidden_channels=32)'

    head_index = torch.tensor([0, 2, 4, 6, 8])
    rel_type = torch.tensor([0, 1, 2, 3, 4])
    tail_index = torch.tensor([1, 3, 5, 7, 9])

    loader = model.loader(head_index, rel_type, tail_index, batch_size=5)
    kge_harness(model, loader)
