# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional
from torch import Tensor
import torch_geometric


class Aggregation(torch_geometric.nn.aggr.Aggregation):
    def assert_sorted_index(self, index: Optional[Tensor]):
        pass
