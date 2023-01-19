# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Any, Generator, Iterable

import torch
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.data.data import BaseData

from poptorch import ICustomArgParser, registerCustomArgParser


class PyGArgsParser(ICustomArgParser):
    @staticmethod
    def sortedTensorKeys(struct) -> Iterable[str]:
        """
        Find all the keys that map to a tensor value in struct. The keys
        are returned in sorted order.
        """
        all_keys = sorted(struct.keys)

        def isTensor(k):
            return isinstance(struct[k], torch.Tensor)

        return filter(isTensor, all_keys)

    def yieldTensors(self, struct) -> Generator[torch.Tensor, None, None]:
        """
        yield every torch.Tensor in struct in sorted order
        """
        for k in self.sortedTensorKeys(struct):
            yield struct[k]

    def reconstruct(self, original_structure, tensor_iterator) -> Any:
        """
        Create a new instance with the same class type as the
        original_structure. This new instance will be initialized with tensors
        from the provided iterator and uses the same sorted keys from the
        yieldTensors() implementation.
        """
        tensor_keys = self.sortedTensorKeys(original_structure)
        kwargs = dict(zip(tensor_keys, tensor_iterator))

        for k in original_structure.keys:
            if k not in kwargs:
                # copy non-tensor properties to the new instance
                kwargs[k] = original_structure[k]

        cls = original_structure.__class__

        if issubclass(cls, Batch):
            assert 'batch' in kwargs.keys(), 'Field `batch` missing'
            assert 'ptr' in kwargs.keys(), 'Field `ptr` missing'
            batch = kwargs.pop('batch')
            ptr = kwargs.pop('ptr')

            data = Data(**kwargs)
            batch_data = Batch.from_data_list([data])
            # We need to recover the 'batch' and 'ptr' tensors, because the
            # ones newly created by 'Batch.from_data_list' have wrong values,
            # as the Data passed to the method represents the whole batch
            # created from a number of separate Data objects. At this stage we
            # do not have any knowledge about each of those separate objects.
            # The same applies to 'num_graphs' attribute.
            batch_data.batch = batch
            batch_data.ptr = ptr
            batch_data._num_graphs = original_structure.num_graphs  # pylint: disable=protected-access
            return batch_data

        return cls(**kwargs)


# PyG uses the BaseData object as the root for data and batch objects
registerCustomArgParser(BaseData, PyGArgsParser())
registerCustomArgParser(type(Batch(_base_cls=Data().__class__)),
                        PyGArgsParser())
registerCustomArgParser(type(Batch(_base_cls=HeteroData().__class__)),
                        PyGArgsParser())
