# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from torch_geometric.data import Batch, Data, HeteroData

DataBatch = type(Batch(_base_cls=Data))
HeteroDataBatch = type(Batch(_base_cls=HeteroData))


def call_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
        return None

    wrapper.has_run = False
    return wrapper
