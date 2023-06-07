# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import functools
import importlib

import torch_geometric
from poptorch_geometric import ops

from poptorch_geometric.common import call_once


class _TorchGeometricOpsSubstitutionManager:

    subsitutions = {
        torch_geometric.nn: {
            "knn_interpolate": ops.knn_interpolate
        },
        torch_geometric.nn.aggr.base.Aggregation: {
            "assert_sorted_index": ops.Aggregation.assert_sorted_index
        },
        torch_geometric.nn.aggr.sort.SortAggregation: {
            "to_dense_batch": ops.Aggregation.to_dense_batch
        },
        torch_geometric.nn.ClusterGCNConv: {
            "forward": ops.ClusterGCNConv.forward
        },
        torch_geometric.nn.conv.edge_conv: {  # pylint: disable=no-member
            "knn": ops.knn
        },
        torch_geometric.nn.conv.gravnet_conv: {  # pylint: disable=no-member
            "knn": ops.knn
        },
        torch_geometric.nn.conv.x_conv: {  # pylint: disable=no-member
            "knn_graph": ops.knn_graph
        },
        torch_geometric.nn.dense.HeteroLinear: {
            "forward": ops.HeteroLinear.forward
        },
        torch_geometric.nn.InstanceNorm: {
            "forward": ops.InstanceNorm.forward
        },
        torch_geometric.nn.conv.MFConv: {
            "forward": ops.MFConv.forward
        },
        torch_geometric.nn.unpool: {
            "knn_interpolate": ops.knn_interpolate
        },
        torch_geometric.nn.pool: {
            "knn": ops.knn,
            "knn_graph": ops.knn_graph,
            "radius": ops.radius,
            "radius_graph": ops.radius_graph,
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
        torch_geometric.experimental.set_experimental_mode(
            True, 'disable_dynamic_shapes')

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

        torch_geometric.experimental.set_experimental_mode(
            False, 'disable_dynamic_shapes')


@call_once
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
            torch_cluster_overrides["knn_graph"] = ops.knn_graph
            torch_cluster_overrides["nearest"] = ops.nearest
            torch_cluster_overrides["radius"] = ops.radius
            torch_cluster_overrides["radius_graph"] = ops.radius_graph


registerOptionalOverrides()
