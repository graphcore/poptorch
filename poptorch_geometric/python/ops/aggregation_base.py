# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple
from torch import Tensor
import torch_geometric
from .to_dense_batch import to_dense_batch


class Aggregation(torch_geometric.nn.aggr.Aggregation):
    def assert_sorted_index(self, index: Optional[Tensor]):
        pass

    def to_dense_batch(
            self,
            x: Tensor,
            index: Optional[Tensor] = None,
            ptr: Optional[Tensor] = None,
            dim_size: Optional[int] = None,
            dim: int = -2,
            fill_value: float = 0.0,
            max_num_elements: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:

        # TODO Currently, `to_dense_batch` can only operate on `index`:
        self.assert_index_present(index)
        self.assert_sorted_index(index)
        self.assert_two_dimensional_input(x, dim)

        return to_dense_batch(
            x,
            index,
            batch_size=dim_size,
            fill_value=fill_value,
            max_num_nodes=max_num_elements,
        )
