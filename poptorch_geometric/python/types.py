# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from itertools import chain
try:
    from functools import singledispatchmethod
except ImportError:
    from singledispatchmethod import singledispatchmethod
from typing import Any, Generator, Union, Iterable, List

import torch
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.data.storage import BaseStorage
from torch_geometric.data.data import BaseData

from poptorch import ICustomArgParser, registerCustomArgParser

from poptorch_geometric.common import DataBatch, HeteroDataBatch, call_once


class PyGArgsParser(ICustomArgParser):
    @staticmethod
    def _sortedTensorKeys(struct: Union[Data, DataBatch]) -> Iterable[str]:
        all_keys = sorted(struct.keys)

        def isTensor(k):
            return isinstance(struct[k], torch.Tensor)

        return filter(isTensor, all_keys)

    @singledispatchmethod
    def yieldTensors(self, struct) -> Generator[torch.Tensor, None, None]:
        raise ValueError(f'Unsupported data type: {type(struct)}')

    @yieldTensors.register
    def _(self, struct: Data
          or DataBatch) -> Generator[torch.Tensor, None, None]:
        for k in self._sortedTensorKeys(struct):
            yield struct[k]

    @yieldTensors.register
    def _(self, struct: HeteroData
          or HeteroDataBatch) -> Generator[torch.Tensor, None, None]:
        def isTensor(val):
            return isinstance(val, torch.Tensor)

        for v in filter(isTensor, struct._global_store.values()):  # pylint: disable=protected-access
            yield v
        for attr in chain(struct.node_stores, struct.edge_stores):
            if isinstance(attr, BaseStorage):
                for v in filter(isTensor, attr.values()):
                    yield v

    @staticmethod
    def _setup_num_fields(
            batch: Union[DataBatch, HeteroDataBatch],
            original_structure: Union[DataBatch, HeteroDataBatch]):
        if hasattr(original_structure, '_num_graphs'):
            batch._num_graphs = original_structure._num_graphs  # pylint: disable=protected-access

        num_nodes = original_structure.num_nodes
        num_edges = original_structure.num_edges
        batch['num_nodes'] = num_nodes
        batch['num_edges'] = num_edges
        if isinstance(batch, HeteroDataBatch):
            # We need to override properties getters, to make them return the
            # proper (device iterations independent) `num_nodes` and `num_edges`
            # The general idea is to return values from `num_nodes` or
            # `num_edges` fields (if defined) in the first place.
            def nodes_fget(sub_self):
                if 'num_nodes' in sub_self._global_store:  # pylint: disable=protected-access
                    return sub_self['num_nodes']
                return super(type(sub_self), sub_self).num_nodes

            setattr(HeteroDataBatch, 'num_nodes', property(fget=nodes_fget))

            def edges_fget(sub_self):
                if 'num_edges' in sub_self._global_store:  # pylint: disable=protected-access
                    return sub_self['num_edges']
                return super(type(sub_self), sub_self).num_edges

            setattr(HeteroDataBatch, 'num_edges', property(fget=edges_fget))

    @staticmethod
    def _add_next(tensor_iterator: Iterable[List[Any]],
                  original_struct_val: Any) -> Any:
        if isinstance(original_struct_val, torch.Tensor):
            return next(tensor_iterator)
        return original_struct_val

    @singledispatchmethod
    def reconstruct(self, original_structure,
                    tensor_iterator: Iterable[torch.Tensor]) -> Any:  # pylint: disable=unused-argument
        raise ValueError(f'Unsupported data type: {type(original_structure)}')

    @reconstruct.register
    def _(self, original_structure: Data or DataBatch,
          tensor_iterator: Iterable[torch.Tensor]) -> Union[Data, DataBatch]:
        """
        Create a new instance with the same class type as the
        original_structure. This new instance will be initialized with tensors
        from the provided iterator and uses the same sorted keys from the
        yieldTensors() implementation.
        """
        tensor_keys = self._sortedTensorKeys(original_structure)

        kwargs = dict()
        for key in tensor_keys:
            kwargs[key] = self._add_next(tensor_iterator,
                                         original_structure[key])

        cls = original_structure.__class__
        if cls is DataBatch:
            batch = Batch(**kwargs, _base_cls=Data)
            self._setup_num_fields(batch, original_structure)
            return batch

        return Data(**kwargs)

    @reconstruct.register
    def _(self, original_structure: HeteroData or HeteroDataBatch,
          tensor_iterator: Iterable[torch.Tensor]
          ) -> Union[HeteroData, HeteroDataBatch]:
        """
        Create a new instance with the same class type as the
        original_structure. This new instance will be initialized with tensors
        from the provided iterator and uses the same sorted keys from the
        yieldTensors() implementation.
        """
        kwargs = dict()

        for key, attr in original_structure._global_store.items():  # pylint: disable=protected-access
            kwargs[key] = self._add_next(tensor_iterator, attr)

        for key, attr in chain(original_structure.node_items(),
                               original_structure.edge_items()):
            if isinstance(attr, BaseStorage):
                kwargs[key] = {
                    k: self._add_next(tensor_iterator, v)
                    for k, v in attr.items()
                }
            else:
                kwargs[key] = self._add_next(attr, attr)

        cls = original_structure.__class__
        if cls is HeteroDataBatch:
            batch = Batch(kwargs, _base_cls=HeteroData)
            self._setup_num_fields(batch, original_structure)
            return batch

        return HeteroData(kwargs)


# PyG uses the BaseData object as the root for data and batch objects.
@call_once
def registerCustomArgParsers():
    registerCustomArgParser(BaseData, PyGArgsParser())
    registerCustomArgParser(DataBatch, PyGArgsParser())
    registerCustomArgParser(HeteroDataBatch, PyGArgsParser())


registerCustomArgParsers()
