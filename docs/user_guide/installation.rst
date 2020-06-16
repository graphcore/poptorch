Installation
------------

PopTorch is part of the Poplar SDK.  It is packaged as a Python wheel
file that can be installed using ``pip``.

PopTorch requires Python 3.6 or later.

For more information about installing the Poplar SDK, see the Getting Started
Guide for your IPU system.

Using a Python virtual environment
__________________________________

We recommend creating a virtual environment to isolate your PopTorch environment
from the system Python environment You can use the Python tool ``virtualenv``
for this. You can create a virtual environment and install PopTorch as shown below:

.. code-block:: bash

    $ virtualenv -p python3 poptorch_test
    $ source poptorch_test/bin/activate
    $ pip install <sdk_path>/poptorch_x.x.x.whl


Setting the environment variables
_________________________________

The PopART and Poplar runtime libraries are required to use PopTorch, so you
will need to set the library search paths, using the scripts provided in the SDK:

.. code-block:: bash

    # Enable the Python environment containing PopTorch (if not already enabled)
    $ source poptorch_test/bin/activate

    # Add the Poplar and PopART runtime libraries to the search path
    $ source <path to poplar installation>/enable.sh
    $ source <path to popart installation>/enable.sh

Validating the setup
____________________

You can run this simple example to verify that the system is working as
expected. This example, and the others in this document can be found in
the Poplar SDK installation.

.. literalinclude:: ../../examples/simple_adder.py
  :language: python
