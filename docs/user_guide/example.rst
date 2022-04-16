Examples
========

You can find PyTorch examples and tutorials in the Graphcore GitHub repositories:

* `PopTorch versions of popular machine learning models for training and inference <https://github.com/graphcore/examples/tree/master/applications/pytorch>`__

* `Tutorials repo <https://github.com/graphcore/tutorials>`__ with:

  - `Tutorials <https://github.com/graphcore/tutorials/tree/sdk-release-2.4/tutorials/pytorch>`__
  - `Examples of PopTorch and IPU features <https://github.com/graphcore/tutorials/tree/sdk-release-2.4/feature_examples/pytorch>`__
  - `Examples of simple models <https://github.com/graphcore/tutorials/tree/sdk-release-2.4/simple_applications/pytorch>`__
  - Source code from videos, blogs and other documents

MNIST example
_____________

The example in :numref:`mnist-example-code` shows how a MNIST model can be run on the IPU. The highlighted lines show the PopTorch-specific code required to run the example on multiple IPUs.

You can download the full source code from GitHub: `mnist.py <https://github.com/graphcore/poptorch/blob/sdk-release-2.4/examples/mnist.py>`__.

To run this example you will need to install the Poplar SDK (see the `Getting Started Guide <https://docs.graphcore.ai/en/latest/getting-started.html>`__ for your IPU system) and the appropriate version of ``torchvision``:

.. code-block:: console

    $ python3 -m pip install torchvision==0.11.1

.. literalinclude:: ../../examples/mnist.py
  :caption: MNIST example
  :name: mnist-example-code
  :start-after: mnist_start
  :end-before: mnist_end
  :emphasize-lines: 12, 15, 17, 20, 35, 96, 99
  :language: python
  :linenos:
  :lineno-match:
