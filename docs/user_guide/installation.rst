============
Installation
============

.. contents::
  :local:

PopTorch is part of the Poplar SDK.  It is packaged as a Python wheel
file that can be installed using ``pip``.

.. important:: pip >= 18.1 is required for PopTorch dependencies to be installed properly.

To update `pip`:

.. code-block:: bash

    $ pip install -U pip

For more information about installing the Poplar SDK, see the relevant
"Getting Started" guide for your IPU system on the Graphcore
`documentation portal <https://docs.graphcore.ai>`_.

Version compatibility
=====================

The following are the corresponding ``torch``, ``torchvision``, ``torchaudio`` and
``torch_scatter`` versions and supported Python versions.

+--------------+-----------+-----------------+----------------+------------------------------+------------+
| ``poptorch`` | ``torch`` | ``torchvision`` | ``torchaudio`` |       ``torch_scatter``      | ``python`` |
+==============+===========+=================+================+==============================+============+
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

Using a Python virtual environment
==================================

We recommend creating a virtual environment to isolate your PopTorch environment
from the system Python environment. You can use the Python tool ``virtualenv``
for this. You can create a virtual environment and install PopTorch as shown below:

.. code-block:: bash

    $ virtualenv -p python3 poptorch_test
    $ source poptorch_test/bin/activate
    $ pip install -U pip
    $ pip install <sdk_path>/poptorch_x.x.x.whl

.. warning:: If, after installing PopTorch, you install a third-party library that requires ``torchvision`` or ``torchaudio`` then that may cause an incompatible version of ``torch`` to be installed.

    To prevent this, after installing PopTorch, use the following commands to install compatible versions of ``torchvision`` or ``torchaudio``:

    .. code-block:: bash

        $ pip install torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
        $ pip install torchaudio==0.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

.. _setting_env:

Setting the environment variables
=================================

The PopART and Poplar runtime libraries are required to use PopTorch, so you
will need to set the library search paths, using the scripts provided in the SDK:

.. code-block:: bash

    # Enable the Python environment containing PopTorch (if not already enabled)
    $ source poptorch_test/bin/activate

    # Add the Poplar and PopART runtime libraries to the search path
    $ source <path to poplar installation>/enable.sh
    $ source <path to popart installation>/enable.sh

Validating the setup
====================

You can run this simple example to verify that the system is working as
expected. This example can be found in the Poplar SDK installation.

.. literalinclude:: ../../examples/simple_adder.py
  :caption: Simple adder example
  :language: python
  :linenos:
