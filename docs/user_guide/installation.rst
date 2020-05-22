Installation
------------

PopTorch is part of the Poplar SDK.  It is packaged inside a python pip wheel
file.

The PopART and Poplar runtime libraries are required to use PopTorch, so the
dynamic library search path must be set up accordingly.

Using a python virtual environment
__________________________________

It is common to isolate python environments from the system python environment
using a tool such as python `virtualenv`.  A typical installation process would
look like this.

.. code-block:: bash

    virtualenv -p python3 poptorch_test
    source poptorch_test/bin/activate
    pip install <sdk_path>/poptorch_x.x.x.whl

To set up the environment to run pytorch scripts, ensure that PopTorch is
available in the python environment, and that Poplar and PopART are both
in the runtime library search path.

.. code-block:: bash

    # Enable the python environment containing PopTorch
    source poptorch_test/bin/activate

    # Add the Poplar and PopART runtime libraries to the search path
    source <path to poplar installation>/enable.sh
    source <path to popart installation>/enable.sh

This simple example can be used to verify that the system is working as
expected.

.. literalinclude:: ../../examples/simple_adder.py
  :language: python
