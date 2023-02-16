# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from itertools import chain
from functools import singledispatchmethod
from typing import Any, Generator, Union, Iterable, List

import torch
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.data.storage import BaseStorage
from torch_geometric.data.data import BaseData

from poptorch import ICustomArgParser, registerCustomArgParser

from .utils import DataBatch, HeteroDataBatch


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
        batch._num_graphs = original_structure._num_graphs  # pylint: disable=protected-access
        batch['num_nodes'] = original_structure.num_nodes
        batch['num_edges'] = original_structure.num_edges
        if isinstance(batch, HeteroDataBatch):
            batch._node_store_dict['num_nodes'] = original_structure.num_nodes  # pylint: disable=protected-access
            batch._edge_store_dict['num_edges'] = original_structure.num_edges  # pylint: disable=protected-access

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
registerCustomArgParser(BaseData, PyGArgsParser())
registerCustomArgParser(DataBatch, PyGArgsParser())
registerCustomArgParser(HeteroDataBatch, PyGArgsParser())
