======================
Optimizing performance
======================

PopTorch Geometric is an extension of PyTorch Geometric allowing models to
fully utilize the IPU hardware and provide the best performance. To achieve
that, PopTorch Geometric uses PopTorch functionality. PopTorch Geometric is
designed in such a way that users can run PyTorch Geometric models with the
least amount of changes to the code and exploit the high performance of IPU
systems.

When working with the IPU, it is always recommended to use fixed-size tensors.
This allows for the static compilation of the Poplar programs and using the same
programs for all the iterations of training and/or inference. This constraint
is not always met when working with Graph Neural Networks because graphs
processed in subsequent iterations can have different numbers of nodes and/or
edges, which results in tensors of different shapes. PopTorch Geometric provides
ways to satisfy this constraint and reach the best performance.

Currently, there are two ways to ensure that all the tensors have fixed
shapes---using either the
`Pad <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.Pad.html#torch_geometric.transforms.Pad>`_
transformation with data loader or the fixed-size data loaders.

.. important:: When working with the IPU, it is required to always use the data
    loader from PopTorch Geometric, either
    :py:class:`poptorch_geometric.dataloader.DataLoader`
    or :py:class:`poptorch_geometric.dataloader.FixedSizeDataLoader`.

All the data loaders in PopTorch Geometric take the `options` argument.
It can be used to set
`PopTorch options <https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/reference.html#poptorch.Options>`_
to process data even more efficiently.

It is recommended to read the
`Efficient data batching <https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/batching.html>`_
chapter of the PopTorch documentation, to understand the possible settings of
the `options` argument.

Pad transformation
==================

`Pad <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.Pad.html#torch_geometric.transforms.Pad>`_
transformation is a graph transformation implemented in PyTorch Geometric. It
sets the fixed number of nodes and edges for all the graphs in the dataset and
pads the node- and edge-level feature tensors so their sizes match the number
of nodes and edges, respectively. Thanks to that, when the data loader creates
a batch of graphs, all the feature tensors of the batch have the same fixed
size and computations can be performed with high efficiency.

A dataset transformed using `Pad` must be used with the
:py:class:`poptorch_geometric.dataloader.DataLoader` data loader to guarantee
compatibility with the IPU.

.. note:: If the dataset you are working on already has a fixed-size feature
    tensors, then using `Pad` transformation is not necessary and it is enough
    to use the :py:class:`poptorch_geometric.dataloader.DataLoader` data
    loader.

Using `Pad` transformation with
:py:class:`poptorch_geometric.dataloader.DataLoader` is recommended when the
graphs in the dataset have a similar number of nodes and edges, so the number
of padding nodes and edges is small.

For examples of usage, refer to :numref:`examples_and_tutorials`.

Fixed-size data loaders
=======================

The alternative method is to use the
:py:class:`poptorch_geometric.dataloader.FixedSizeDataLoader` class with the
dataset without the `Pad` transformation. The data loader uses
:py:class:`poptorch_geometric.collate.FixedSizeCollater` underneath to
create mini-batches of graphs with a fixed number of nodes and edges from the
initial graphs that do not necessarily have the same number of nodes and edges.
The data loader combines graphs from the dataset and creates dummy graphs such
that the whole mini-batch has a fixed number of nodes, edges and graphs.

By default the `FixedSizeStrategy.PadToMax` strategy is used, which pads the
mini-batches to a fixed-size where the resulting mini-batches have a fixed
number of samples in each mini-batch and one padding graph at the end of the
mini-batch.

The data loader can also produce packed batches with a variable number of
graphs in each mini-batch. This can help reduce the amount of space in each
mini-batch assigned to padding. This is enabled by using
`FixedSizeStrategy.StreamPack` which changes the underlying sampler to
:py:class:`poptorch_geometric.stream_packing_sampler.StreamPackingSampler`.
In this case, each mini-batch contains a certain number of dummy graphs, so
that the total number of graphs in the mini-batch is constant.

Compared to `Pad` transformation, instead of padding each sample in the batch,
the data loader pads the entire batch, which is often more efficient and the
created batches are easier to manage since all the padding nodes and edges are
at the end.

For examples of usage, refer to :numref:`examples_and_tutorials`.
