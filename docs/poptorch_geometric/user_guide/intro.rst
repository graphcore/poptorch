============
Introduction
============

PopTorch Geometric is a set of extensions for PyTorch Geometric, enabling Graph
Neural Network models to be trained, evaluated and used on Graphcore IPU
hardware.
PopTorch Geometric has been designed to require as few changes as possible to
your models to run on the IPU.
However, it does have some differences from native PyTorch Geometric execution,
in order to get the most out of IPU hardware.
PopTorch Geometric depends on the functionality provided by PopTorch.

See the "Getting Started" guide for your IPU system on the Graphcore
`documentation portal <https://docs.graphcore.ai/>`_ for information on
installing the Poplar SDK, PopTorch and PopTorch Geometric.
