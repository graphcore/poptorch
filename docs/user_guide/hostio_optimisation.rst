=====================
Efficient IPU I/O
=====================

When developing applications for the IPU, maximising the I/O performance is
important. If an application is I/O-bound, after optimisation of the host data
loading, then you can explore further optimisations of the movement of data
into the IPU. This chapter will cover two options that can improve I/O
performance.

Prefetch and Multibuffering
===========================

Poplar supports prefetching and multibuffering to improve the I/O performance.
For more details, see the `Poplar and PopLibs User Guide <https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/poplar_programs.html#data-streams-and-remote-buffers>`__.

Prefetch is enabled by default in poplar. The default buffer depth is 1. You
can change it to a higher value to improve I/O performance:

.. code-block:: python

    opts = poptorch.Options()
    opts._Popart.set("defaultPrefetchBufferingDepth", 3)

Using multibuffering is specially useful when you see large ``StreamCopyBegin``
or ``StreamCopyEnd`` in your application's profile.

For example, :numref:`figNoBuffering` shows a profile of a simple pogram
without using buffering. The program consists of a loop where the IPU gets data
from the host, processes it and sends the result back. The ``StreamCopy``,
in light orange, represents the data transfer. The first one is the host to IPU
transfer, the second one is the IPU to host transfer. They are split into a
``Begin``, a ``Mid`` and an ``End`` phase. In the ``Begin`` and ``End`` phases,
the IPU waits for the host to become ready. In the ``Mid`` phase the IPU
perfoms the transfer. Between the ``StreamCopy`` operations, are the compute
steps in red. In this profile, you can see the IPU waiting for the host a
significant amount of time.

.. figure:: no-buffering-profile.png
  :name: figNoBuffering
  :width: 100%

  Profile with multibuffering disabled

:numref:`figWithBuffering` shows the profile of the same program with
buffering. You can see that the IPU no longer waits for the host: the ``Begin``
and ``End`` section of the ``StreamCopy`` are gone.

.. figure:: with-buffering-profile.png
  :name: figWithBuffering
  :width: 100%

  Profile with multibuffering enabled and related improvements

Overlaping compute and I/O
==========================

To optimise the I/O further, you can dedicate some tile to the
communication and let the rest of the tiles compute. The computation
time will be adversly affected by having access to less tiles, so there is a
tradeoff between optimising I/O and optimising compute here.

To overlap compute and I/O, a number of things must be done. First, in the
PopTorch options, you must specify the number of I/O tiles and select
one of ``ShardedExecution``, ``ParallelPhasedExecution`` or
``SerialPhasedExecution`` as the ``ExecutionStrategy``:

.. code-block:: python

    opts.TensorLocations.numIOTiles(64)
    opts.setExecutionStrategy(poptorch.ShardedExecution())

Second, in the forward method of the model, you must set the ``OverlapMode``
for the inputs and ouputs of the model to ``OverlapDeviceIterationLoop``, as
follows:

.. code-block:: python

    def forward(self, x):
      x = poptorch.set_overlap_for_input(x, poptorch.OverlapMode.OverlapDeviceIterationLoop)
      x = some_compute(x)
      x = poptorch.set_overlap_for_output(x, poptorch.OverlapMode.OverlapDeviceIterationLoop)
      return x

:numref:`figWithBufferingOverlap` shows the profile of our simple program with both
compute I/O overlap and multibuffering enabled. The compute (in red) and the
I/O (in orange) are stacked since they happen at the same time.

.. _figWithBufferingOverlap:
.. figure:: with-buffering-overlap-profile.png

  Profile with both multibuffering and I/O compute overlap enabled and related improvements
