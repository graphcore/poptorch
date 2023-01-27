# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import os.path as osp

from torch_geometric import seed_everything
from torch_geometric.datasets import Entities
from torch_geometric.datasets import FakeDataset as FDS
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import Compose, GCNNorm, NormalizeFeatures


class DataSets:
    def __init__(self, root):
        self.root = root

    def Cora(self):
        return Planetoid(osp.join(self.root, 'Cora'), 'Cora')

    def CiteSeer(self):
        return Planetoid(osp.join(self.root, 'CiteSeer'), 'CiteSeer')

    def PubMed(self):
        return Planetoid(osp.join(self.root, 'PubMed'), 'PubMed')

    def mutag(self):
        return Entities(osp.join(self.root, 'EntitiesMUTAG'), 'mutag')

    def FakeDataset(self):
        seed_everything(0)

        transform = Compose([GCNNorm(), NormalizeFeatures()])

        dataset = FDS(
            num_graphs=1000,
            avg_num_nodes=16,
            avg_degree=5,
            transform=transform,
            num_channels=64,
        )
        setattr(dataset, 'name', 'FakeDataset')
        return dataset
