# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from torch_geometric.loader.dataloader import Collater as PyGCollater


# TODO: Upstream that change (default arguments) to PyG when upstreaming
# DataLoaders.
class Collater(PyGCollater):
    def __init__(self, follow_batch=None, exclude_keys=None):
        follow_batch = follow_batch or []
        exclude_keys = exclude_keys or []
        super().__init__(follow_batch, exclude_keys)
