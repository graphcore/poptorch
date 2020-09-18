=====================
Experimental features
=====================

Distributed execution
=====================

PopTorch supports distributed execution on IPU-POD using the IPU over Fabric
(IPUoF).

You must create a partition using either ``vipu-cli`` before running
PopTorch, or directly using the :py:class:`poptorch.distributed.VirtualIpuManager`
python wrapper in your script.

The only change to your code needed is to set the id of the current process and
the total number of processes the execution is distributed across using
:py:meth:`~poptorch.options._DistributedOptions.configureProcessId` 

.. literalinclude:: device_iterations.py
  :caption: Changes required for distributed execution
  :lines: 163-198
  :emphasize-lines: 9, 12, 18 
  :linenos:

.. note:: The ``DataLoader`` will automatically select a different subset of the
  dataset based on the process id.

.. warning:: All the processes must use the same seed if ``shuffle=True`` is used
  for the ``DataLoader``.

Distributed execution using Open MPI
------------------------------------

In the ``__main__`` part of your script, simply forward the rank of the
process and the world size to the ``process`` function:

.. code-block:: python

  import os

  if __name__ == "__main__":
    process(
      process_id=int(os.environ.get("OMPI_COMM_WORLD_RANK",0)),
      num_processes=int(os.environ.get("OMPI_COMM_WORLD_SIZE",1))
    )

Use ``vipu-cli`` to create a partition then use ``mpirun`` to start
the processes:

.. code-block:: bash
  
   # Create a partition
   vipu-cli create partition my_partition --size 4 --num-gcds 2 --gcd-sync-replicas 4
   # Start 2 processes
   mpirun -np 2 myScript.py

Distributed execution using Python
----------------------------------

.. autoclass:: poptorch.distributed.VirtualIpuManager
   :members:

.. literalinclude:: device_iterations.py
  :caption: How to start the processes from python
  :lines: 202-232
  :linenos:

