# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import functools
import importlib

import torch_geometric
import torch_cluster
from poptorch_geometric import ops


class _TorchGeometricOpsSubstitutionManager:

    subsitutions = {
        torch_cluster: {
            "knn": ops.knn
        },
        torch_geometric.nn.conv.edge_conv: {  # pylint: disable=no-member
            "knn": ops.knn
        },
        torch_geometric.nn.conv.gravnet_conv: {  # pylint: disable=no-member
            "knn": ops.knn
        },
        torch_geometric.nn: {
            "knn_interpolate": ops.knn_interpolate
        },
        torch_geometric.nn.unpool: {
            "knn_interpolate": ops.knn_interpolate
        },
        torch_geometric.nn.ClusterGCNConv: {
            "forward": ops.ClusterGCNConv.forward
        }
    }

    def __init__(self):
        self.overrides = {}

    def __enter__(self):
        self.replace()
        return self

    def __exit__(self, exc_type, value, traceback):
        self.restore()

    def replace(self):
        def create_wrapper(f, replacement_f):
            @functools.wraps(f)
            def _wrapper(*args, **kwargs):
                return replacement_f(*args, **kwargs)

            return _wrapper

        for mod, replacement_map in self.subsitutions.items():
            for op_name, replacement in replacement_map.items():
                func = getattr(mod, op_name)
                self.overrides.setdefault(mod, {})[op_name] = func
                setattr(mod, op_name, create_wrapper(func, replacement))

    def restore(self):
        for mod, replacement_map in self.overrides.items():
            for op_name, func in replacement_map.items():
                setattr(mod, op_name, func)


def registerOptionalOverrides():
    torch_cluster_spec = importlib.util.find_spec("torch_cluster")
    if torch_cluster_spec is not None:
        loader = torch_cluster_spec.loader
        if loader is not None:
            torch_cluster = loader.load_module()
            torch_cluster_overrides = \
                _TorchGeometricOpsSubstitutionManager.subsitutions.setdefault(
                    torch_cluster, {})
            torch_cluster_overrides["knn"] = ops.knn


registerOptionalOverrides()
