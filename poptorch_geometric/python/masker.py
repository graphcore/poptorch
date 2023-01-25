# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
Provides an interface that reduces coupling between padding which
happens in the dataloader and the masking which needs to happen in
the model.

The idea is fairly simple: the dataloader defines the masking strategy
for nodes, edges, and graphs. The IPU GNNs consume that interface, and
it is easy to make the mask operations no-ops for compatibility with other
hardware.

### Expected usage pattern

```python
import torch_geometric as pyg
from torch import nn
import poptorch

class IpuGNN(pyg.SomeGNN):

    def __init__(self, masker: Masker = NoMasker(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masker = masker  # New line to support this pattern
        self.loss = nn.CrossEntropyLoss()

    def forward(self, node_mask, y, *args, **kwargs):
        '''Common poptorch usage pattern of needing to put the loss in the
        forward'''
        node_prediction = super().forward(*args, **kwargs)
        # clear interface for model code to program to
        masked_pred = self.masker.node_masker(node_prediction, node_mask)
        if self.training:
            return masked_pred, self.loss(y, masked_pred)
        return masked_pred


options = poptorch.Options()
dataloader = poptorch_geometric.create_dataloader(
    dataset=dataset,
    num_nodes=6000,
    options=options,
    fixed_size=True,
    collater_args={
        "num_edges": 12000,
    },
)

model = IpuGNN(dataloader.masker)
train_model = poptorch.TrainingModel(model, options=options, ...)

for data in dataloader:
    # Need to pass the mask as an extra argument.
    train_model(data.node_mask, data.y, ...)
```

### Expected benefit

The big benefit is it pushes the responsibility of writing the masking
functions to the same piece of code that also implements the padding and
generates the node mask.

It means consumers of a dataloader don't have to worry about implementation
details.
"""
import abc
from typing import Callable, Optional, Tuple, Union

import torch

Entries = Union[torch.Tensor, Tuple[torch.Tensor, ...]]
Mask = Optional[torch.Tensor]
Layer = Callable[[torch.Tensor], torch.Tensor]
DecoratedLayer = Callable[[torch.Tensor], torch.Tensor]


class Masker(abc.ABC):
    """
    The masker provides a way to decouple the model from the
    implementation of the dataloading. We provide a stable interface
    for masking padded data and graphs.

    Dataloaders that implement padding should also generates masking functions
    for you by either implementing this :class:`Masker` interface or by
    composing a `layer_mask` attribute to the class. Models which are
    compatible can then use those masks as intermediate layers before the loss
    or before pooling operations to avoid the back propagation:

    ```python
    class Net(Module):
        def __init__(self, layer_mask):
            self.node_layer = pyg.GraphConv()
            self.masker = layer_mask
            self.loss = nn.loss()

        def forward(self, x, y, mask):
            x = self.node_layer(x)
            x = self.node_layer(x)
            x = self.node_layer(x)
            pred = self.masker.node_masker(x, mask=mask)
            return loss(y, pred)
    ```

    By implementing this interface we let the user change their dataloading
    Pipeline without having to go into the code of model.

    Note:
        Code in the node, edge and graph masker will be run on the IPU and
        needs to be compatible with torch.jit.trace.
    """

    @abc.abstractmethod
    def node_masker(self, node_entries: Entries, mask: Mask = None) -> Entries:
        """Masks out nodes which were added by padding/batching/clustering"""

    @abc.abstractmethod
    def edge_masker(self, edge_entries: Entries, mask: Mask = None) -> Entries:
        """Masks out edges which were added by padding/batching/clustering"""

    @abc.abstractmethod
    def graph_masker(self, graph_entries: Entries,
                     mask: Mask = None) -> Entries:
        """Masks out graphs which were added by padding/batching/clustering"""


class NoMasker(Masker):
    """A null op masker to give when masking is unnecessary"""

    def node_masker(self, node_entries: Entries, mask: Mask = None) -> Entries:
        return node_entries

    def edge_masker(self, edge_entries: Entries, mask: Mask = None) -> Entries:
        return edge_entries

    def graph_masker(self, graph_entries: Entries,
                     mask: Mask = None) -> Entries:
        return graph_entries


class LayerMasker(abc.ABC):
    """
    The layer masker provides a way to decouple the model from the
    implementation of the dataloading. We provide a stable interface
    for masking layers which need to operated on padded data and graphs.

    Note:
        This is an alternative proposal to the :class:`Masker` above. It
        differs by proposing we use decoration of the layers instead of
        calling in between the layers.

        The decoration approach might help handle cases where a lot of
        masking is necessary by decorating layers defined in the
        `__init__` of a `Module` removing the need for changing the
        forward method.

    This default implementation is sufficient for layers which only take
    tensors that will be masked according to the same attribute (node, edge,
    or graph) this will not handle a layer which needs two tensors one related
    to edges and one related to nodes.
    """

    def __init__(self, masker: Masker) -> None:
        super().__init__()
        self.masker = masker

    @abc.abstractmethod
    def node_masker(self, layer: Layer) -> DecoratedLayer:
        def masked_layer(*args, mask=None):
            return layer(*self.masker.node_masker(args, mask=mask))

        return masked_layer

    @abc.abstractmethod
    def edge_masker(self, layer: Layer) -> DecoratedLayer:
        def masked_layer(*args, mask=None):
            return layer(*self.masker.edge_masker(args, mask=mask))

        return masked_layer

    @abc.abstractmethod
    def graph_masker(self, layer: Layer) -> DecoratedLayer:
        def masked_layer(*args, mask=None):
            return layer(*self.masker.graph_masker(args, mask=mask))

        return masked_layer


class PreLayerMasker(LayerMasker):
    """Simplest Layer masker"""

    # pylint: disable=useless-super-delegation
    def node_masker(self, layer: Layer) -> DecoratedLayer:
        return super().node_masker(layer)

    def edge_masker(self, layer: Layer) -> DecoratedLayer:
        return super().edge_masker(layer)

    def graph_masker(self, layer: Layer) -> DecoratedLayer:
        return super().graph_masker(layer)
