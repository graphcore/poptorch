Features
--------

The main classes provided by PopTorch are:

.. py:function:: trainingModel(model, device_iterations, gradient_accumulation=1, profile=False, trace_model=True)

    Create a PopTorch training model, from a PyTorch model, to run on IPU hardware.

    :param nn.Module model: A PyTorch model.
    :param int device_iterations:
    :param int gradient_accumulation:
    :param bool profile: Enable generation of profiling information by Poplar.
    :param bool trace_model: Compile model before running.

.. py:function:: inferenceModel(model, device_iterations=1, profile=False, trace_model=True)

    Create a PopTorch inference model, from a PyTorch model, to run on IPU hardware.

    :param nn.Module model: A PyTorch model.
    :param device_iterations:
    :param profile: Enable generation of profiling information by Poplar.
    :param trace_model: Compile model before running.

.. py:class:: IPU(ipu_id, layer_to_call=None)

    Runs a layer on a specified IPU.

    :param int ipu_id: The id of the IPU to run on.
    :param layer_to_call: The layer to run on the specified IPU.

