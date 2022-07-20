Examples
========

You can find PyTorch examples and tutorials in the Graphcore GitHub repositories:

* `Examples of popular machine learning models for training and inference <https://github.com/graphcore/examples>`_

* `Tutorials repo <https://github.com/graphcore/tutorials>`_ with:

  - :tutorials-repo:`Tutorials <tutorials/pytorch>`
  - :tutorials-repo:`Examples of PopTorch and IPU features <feature_examples/pytorch>`
  - :tutorials-repo:`Examples of simple models <simple_applications/pytorch>`
  - Source code from videos, blogs and other documents

MNIST example
_____________

The example in :numref:`mnist-example-code` shows how a MNIST model can be run on the IPU. The highlighted lines show the PopTorch-specific code required to run the example on multiple IPUs.

You can download the full source code from GitHub: `mnist.py <https://github.com/graphcore/poptorch/blob/sdk-release-2.4/examples/mnist.py>`_.

To run this example you will need to install the Poplar SDK (see the `Getting Started Guide <https://docs.graphcore.ai/en/latest/getting-started.html>`_ for your IPU system) and the appropriate version of ``torchvision``:

.. code-block:: console

    $ python3 -m pip install torchvision==0.11.1

.. literalinclude:: ../../examples/mnist.py
  :caption: MNIST example
  :name: mnist-example-code
  :start-after: mnist_start
  :end-before: mnist_end
  :emphasize-lines: 12, 15, 17, 20, 35, 96, 99
  :language: python
  :dedent: 3
  :linenos:
  :lineno-match:
