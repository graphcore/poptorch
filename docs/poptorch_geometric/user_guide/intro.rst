============
Introduction
============

.. admonition:: Experimental Release

    This is an experimental release of PopTorch Geometric. Not all features of PyTorch Geometric are supported, and some functions may not work as expected. The implementation may change without warning in future releases in ways that are not backwards compatible.

PopTorch Geometric is a set of extensions for PyTorch Geometric, enabling Graph
Neural Network models to be trained, evaluated and used on Graphcore IPU
hardware.

PopTorch Geometric has been designed to require as few changes as possible to
your models to run on the IPU.
However, it does have some differences from native PyTorch Geometric execution,
in order to get the most out of IPU hardware.

PopTorch Geometric depends on the functionality provided by PopTorch.

PopTorch and PopTorch Geometric are included in the `Poplar SDK <https://docs.graphcore.ai/projects/sdk-overview/>`__. See the `Getting Started guide <https://docs.graphcore.ai/en/latest/getting-started.html#getting-started>`_ for your system for how to
install the Poplar SDK. Refer to :numref:`installation` for how to install the PopTorch and PopTorch Geometric wheels.
