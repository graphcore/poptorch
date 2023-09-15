.. _installation:

============
Installation
============

.. contents::
  :local:

PopTorch is included with the Poplar SDK (see the `Getting Started guide <https://docs.graphcore.ai/en/latest/getting-started.html>`_ for your system for how to install the Poplar SDK.).  PopTorch is packaged as a Python wheel
file that can be installed using ``pip``.

.. important:: pip >= 18.1 is required for PopTorch dependencies to be installed properly.

To update ``pip``:

.. code-block:: bash

    $ pip install -U pip


Version compatibility
=====================

The following are the corresponding ``torch``, ``torchvision``, ``torchaudio`` and
``torch_scatter`` versions and supported Python versions.

+--------------+-----------+-----------------+----------------+------------------------------+------------+
| ``poptorch`` | ``torch`` | ``torchvision`` | ``torchaudio`` |       ``torch_scatter``      | ``python`` |
+==============+===========+=================+================+==============================+============+
|     3.3      |   2.0.1   |      0.15.2     |      2.0.1     |   >=2.0.9 and <=2.1.1        |    >=3.8   |
+--------------+-----------+-----------------+----------------+------------------------------+------------+
|     3.2      |   1.13.1  |      0.14.1     |      0.13.1    |   >=2.0.9 and <=2.1.0        |    >=3.7   |
+--------------+-----------+-----------------+----------------+------------------------------+------------+
|     3.1      |   1.13.0  |      0.14.0     |      0.13.0    |   >=2.0.9 and <=2.1.0        |    >=3.7   |
+--------------+-----------+-----------------+----------------+------------------------------+------------+
|     3.0      |   1.10.0  |      0.11.1     |      0.10.0    |             N/A              |    >=3.6   |
+--------------+-----------+-----------------+----------------+------------------------------+------------+
|     2.6      |   1.10.0  |      0.11.1     |      0.10.0    |             N/A              |    >=3.6   |
+--------------+-----------+-----------------+----------------+------------------------------+------------+
|     2.5      |   1.10.0  |      0.11.1     |      0.10.0    |             N/A              |    >=3.6   |
+--------------+-----------+-----------------+----------------+------------------------------+------------+
|     2.4      |   1.10.0  |      0.11.1     |      0.10.0    |             N/A              |    >=3.6   |
+--------------+-----------+-----------------+----------------+------------------------------+------------+
|     2.3      |   1.9.0   |      0.10.0     |      0.9.0     |             N/A              |    >=3.6   |
+--------------+-----------+-----------------+----------------+------------------------------+------------+
|     2.2      |   1.9.0   |      0.10.0     |      0.9.0     |             N/A              |    >=3.6   |
+--------------+-----------+-----------------+----------------+------------------------------+------------+
|     2.1      |   1.7.1   |      0.8.2      |      0.7.1     |             N/A              |    >=3.6   |
+--------------+-----------+-----------------+----------------+------------------------------+------------+
|     2.0      |   1.7.1   |      0.8.2      |      0.7.1     |             N/A              |    >=3.6   |
+--------------+-----------+-----------------+----------------+------------------------------+------------+
|     1.4      |   1.6.0   |      0.7.0      |      0.6.0     |             N/A              |    >=3.6   |
+--------------+-----------+-----------------+----------------+------------------------------+------------+

Based on https://github.com/pytorch/vision/blob/master/README.md

.. note:: To ensure version compatibility, ``torchvision`` and ``torchaudio`` are automatically installed with PopTorch in Poplar SDK 3.3 and later.


Using a Python virtual environment
==================================

We recommend creating and activating a virtual environment to isolate your PopTorch environment
from the system Python environment. You can use the Python tool ``virtualenv``
for this. You can create a virtual environment and install PopTorch as shown below:

.. code-block:: bash

    $ virtualenv -p python3 poptorch_test
    $ source poptorch_test/bin/activate
    $ pip install -U pip
    $ pip install <sdk_path>/poptorch_x.x.x.whl


.. _setting_env:

Setting the environment variables
=================================

The PopART and Poplar runtime libraries are required to use PopTorch, so you
will need to set the library search paths, using the scripts provided in the SDK:

.. code-block:: bash

    # Enable the Python environment containing PopTorch (if not already enabled)
    $ source poptorch_test/bin/activate

    # Add the Poplar and PopART runtime libraries to the search path
    $ source <sdk_path>/poplar-ubuntu_<os_ver>-<poplar_ver>+<build>/enable.sh
    $ source <sdk_path>/popart-ubuntu_<os_ver>-<poplar_ver>+<build>/enable.sh

where ``<sdk_path>`` is the location of the Poplar SDK on your system. ``<os_ver>`` is the version of Ubuntu on your system, ``<poplar_ver>`` is the software version number of the Poplar SDK and ``<build>`` is the build information.


Validating the setup
====================

You can run this simple example to verify that the system is working as
expected. This example can be found in the Poplar SDK ``examples`` directory.

.. literalinclude:: ../../examples/simple_adder.py
  :caption: Simple adder example
  :language: python
  :linenos:
