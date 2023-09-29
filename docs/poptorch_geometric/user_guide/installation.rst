.. _installation:

============
Installation
============

PopTorch Geometric is included as part of the Poplar SDK (see the `Getting
Started guide
<https://docs.graphcore.ai/en/latest/getting-started.html#getting-started>`_ for
your system for how to install the Poplar SDK. ). PopTorch Geometric is packaged
as a Python wheel file that can be installed using ``pip``. PopTorch Geometric
requires the installation of PopTorch, which is also a part of the Poplar SDK.

To use PopTorch Geometric you must first install the PopTorch wheel
and then the PopTorch Geometric wheel. All the necessary
dependencies (including ``torch`` and ``pytorch_geometric``) will be installed
automatically.


.. important:: pip >= 18.1 is required for PopTorch dependencies to be
    installed properly.

To update `pip`:

.. code-block:: bash

    $ pip install -U pip


Version compatibility
~~~~~~~~~~~~~~~~~~~~~

PopTorch Geometric and PopTorch wheels should always come from the same Poplar
SDK version to guarantee version compatibility.

The following are the corresponding ``poptorch``, ``torch``, ``torchvision`` and ``torchaudio``
versions and supported Python versions.

+------------------------+-----------------------+-------------+-----------------+----------------+------------+
| ``poptorch_geometric`` | ``pytorch_geometric`` |  ``torch``  | ``torchvision`` | ``torchaudio`` | ``python`` |
+========================+=======================+=============+=================+================+============+
|          3.3           |   2.4.0.dev20230613   |    2.0.1    |      0.15.2     |      2.0.1     |   >= 3.8   |
+------------------------+-----------------------+-------------+-----------------+----------------+------------+
|          3.2           |   2.3.0.dev20230222   |    1.13.1   |      0.14.1     |      0.13.1    |   >= 3.7   |
+------------------------+-----------------------+-------------+-----------------+----------------+------------+

.. note:: To ensure version compatibility, ``torchvision`` and ``torchaudio`` are automatically installed with PopTorch in Poplar SDK 3.3 and later.

Installation using Python virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend creating and activating a virtual environment to isolate your
PopTorch Geometric environment from the system Python environment. You can use
the Python ``virtualenv`` tool for this.

.. code-block:: bash

    $ virtualenv -p python3 poptorch_test
    $ source poptorch_test/bin/activate

After activating the virtual environment, you need to first install the PopTorch wheel.

.. code-block:: bash

    $ pip install <sdk_path>/poptorch-x.x.x.whl

where ``<sdk_path>`` is the location of the Poplar SDK on your system.

See the
`PopTorch installation guide <https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/installation.html>`_
for more information on installing the PopTorch wheel.

Then, install the PopTorch Geometric wheel:

.. code-block:: bash

    # Enable the Python environment containing PopTorch (if not already enabled)
    $ source poptorch_test/bin/activate
    $ pip install <sdk_path>/poptorch_geometric-x.x.x.whl

where ``<sdk_path>`` is the location of the Poplar SDK on your system.

Setting the environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PopART and Poplar runtime libraries are required to use PopTorch Geometric,
so you will need to set the library search paths, using the scripts provided
in the SDK:

.. code-block:: bash

    # Enable the Python environment containing PopTorch (if not already enabled)
    $ source poptorch_test/bin/activate

    # Add the Poplar and PopART runtime libraries to the search path
    $ source <sdk_path>/poplar-ubuntu_<os_ver>-<poplar_ver>+<build>/enable.sh
    $ source <sdk_path>/popart-ubuntu_<os_ver>-<poplar_ver>+<build>/enable.sh

where ``<sdk_path>`` is the location of the Poplar SDK on your system, ``<os_ver>`` is the version of Ubuntu on your system, ``<poplar_ver>`` is the software version number of the Poplar SDK and ``<build>`` is the build information.


Validating the setup
~~~~~~~~~~~~~~~~~~~~

In order to validate that everything is installed correctly in your
environment, you can run the following commands and see if they execute without
an exception and the displayed version matches the packages that you installed:

.. code-block:: bash

    $ python -c "import poptorch;print(poptorch.__version__)"
    $ python -c "import poptorch_geometric;print(poptorch_geometric.__version__)"
