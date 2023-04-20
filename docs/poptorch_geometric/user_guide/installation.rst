============
Installation
============

PopTorch Geometric is part of the Poplar SDK. It is packaged as a Python wheel
file that can be installed using `pip`. PopTorch Geometric requires the
installation of PopTorch, which is also part of the Poplar SDK.

.. important:: pip >= 18.1 is required for PopTorch dependencies to be
    installed properly.

To update `pip`:

.. code-block:: bash

    $ pip install -U pip


Version compatibility
~~~~~~~~~~~~~~~~~~~~~

PopTorch Geometric and PopTorch wheels should always come from the same Poplar
SDK version to guarantee version compatibility.

The following is the corresponding ``poptorch``, ``torch`` and ``torchvision``
versions and supported Python versions.

+------------------------+-----------------------+-------------+-----------------+------------+
| ``poptorch_geometric`` | ``pytorch_geometric`` |  ``torch``  | ``torchvision`` | ``python`` |
+========================+=======================+=============+=================+============+
|          3.2           |   2.3.0.dev20230222   |    1.13.1   |      0.14.1     |   >= 3.7   |
+------------------------+-----------------------+-------------+-----------------+------------+


Installation using Python virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend creating a virtual environment to isolate your PopTorch
environment from the system Python environment. You can use the Python
``virtualenv`` tool for this.

.. code-block:: bash

    $ virtualenv -p python3 poptorch_test
    $ source poptorch_test/bin/activate

After creating the virtual environment, you need to install the PopTorch wheel.

.. code-block:: bash

    $ pip install <sdk_path>/poptorch-x.x.x.whl

See the
`PopTorch installation guide <https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/installation.html>`_
for more information on installing the PopTorch wheel.

Then the PopTorch Geometric wheel can be installed using the commands below.

.. code-block:: bash

    # Enable the Python environment containing PopTorch (if not already enabled)
    $ source poptorch_test/bin/activate
    $ pip install <sdk_path>/poptorch_geometric-x.x.x.whl


Setting the environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PopART and Poplar runtime libraries are required to use PopTorch Geometric,
so you will need to set the library search paths, using the scripts provided
in the SDK:

.. code-block:: bash

    # Enable the Python environment containing PopTorch (if not already enabled)
    $ source poptorch_test/bin/activate

    # Add the Poplar and PopART runtime libraries to the search path
    $ source <path to poplar installation>/enable.sh
    $ source <path to popart installation>/enable.sh

To use PopTorch Geometric it is required to first install the PopTorch wheel
and install the PopTorch Geometric wheel afterward. All the necessary
dependencies (including ``torch`` and ``pytorch_geometric``) will be installed
automatically.


Validating the setup
~~~~~~~~~~~~~~~~~~~~

In order to validate that everything is installed correctly in your
environment, you can run the following commands and see if they execute without
an exception and the displayed version matches the packages that you installed:

.. code-block:: bash

    $ python -c "import poptorch;print(poptorch.__version__)"
    $ python -c "import poptorch_geometric;print(poptorch_geometric.__version__)"
