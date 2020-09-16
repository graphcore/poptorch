# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
from .logging import logger

begin_ipu_block = torch.ops.poptorch.begin_ipu_block
end_ipu_block = torch.ops.poptorch.end_ipu_block
ipu_print_tensor = torch.ops.poptorch.ipu_print_tensor
set_available_memory = torch.ops.poptorch.set_available_memory
nop = torch.ops.poptorch.nop


class IPU(torch.nn.Module):
    """Runs a layer on a specified IPU.

    All layers after this layer will also run on
    the same IPU until another IPU wrapper is encountered.

    The execution will be "pipelined" where each IPU is executing one stage
    of the operation, as the previous IPU is executing a previous stage on
    the next batch and subsequent IPUs are executing subsequent stages on
    previous batches.

    Can be used as either a scope variable:

    >>> with poptorch.IPU(1):
    ...     self.layer = MyLayer(x)

    Or as a wrapper:

    >>> self.layer = poptorch.IPU(1, MyLayer(x))
    """

    def __init__(self, ipu_id, layer_to_call=None):
        """
        :param int ipu_id: The id of the IPU to run on. All subsequent layers
                           of the network will run on this IPU until another
                           layer is wrapped. By default all layers will be on
                           IPU 0 until the first pipeline annotation is
                           encountered. Note that the ``ipu_id`` is an index
                           in a multi-IPU device within PopTorch, and is
                           separate and distinct from the device ids used by
                           ``gc-info``.

        :param layer_to_call: The layer to run on the specified IPU.
        """

        super().__init__()
        logger.warning(
            """IPU annotations are going to be deprecated in favour of
                        beginPhase annotations.""")

        self.ipu_id = ipu_id
        self.layer_to_call = layer_to_call

    def __enter__(self):

        begin_ipu_block(self.ipu_id, -1)

    def __exit__(self, type, value, traceback):
        end_ipu_block()

    def __call__(self, *input, **kwargs):
        begin_ipu_block(self.ipu_id, -1)
        out = self.layer_to_call(*input, **kwargs)
        return out


class Phase(torch.nn.Module):
    """Runs a layer on a specified IPU.

    All layers after this layer will also run on
    the same IPU and Phase until another Phase is encountered.

    By default this will be "pipelined" execution, however this
    can be overridden by the popart session option.

    >>> with poptorch.Phase(1):
    ...     self.layer = MyLayer(x)

    """

    def __init__(self, ipu_id, phase_id=-1):
        """
        :param int ipu_id: The id of the IPU to run on. All subsequent layers
                           of the network will run on this IPU until another
                           layer is wrapped. By default all layers will be on
                           IPU 0 until the first pipeline annotation is
                           encountered. Note that the ``ipu_id`` is an index
                           in a multi-IPU device within PopTorch, and is
                           separate and distinct from the device ids used by
                           ``gc-info``.
        :param int phase_id: The PopART execution phase this code block should
                           belong on.
        """
        super().__init__()
        self.ipu_id = ipu_id
        self.phase_id = phase_id

    def __enter__(self):
        begin_ipu_block(self.ipu_id, self.phase_id)

    def __exit__(self, type, value, traceback):
        end_ipu_block()


class BeginPhase(torch.nn.Module):
    """Runs a layer on a specified Phase mapped to a specific API.

    All layers after this layer will also run on
    the same IPU until another IPU wrapper is encountered.

    By default this will be "pipelined" execution, however this
    can be overridden by the popart session option.

    >>> self.layer = poptorch.Phase(1, MyLayer(x), phase_id=1)

    """

    def __init__(self, ipu_id, layer_to_call, phase_id=-1):
        """
        :param int ipu_id: The id of the IPU to run on. All subsequent layers
                           of the network will run on this IPU until another
                           layer is wrapped. By default all layers will be on
                           IPU 0 until the first pipeline annotation is
                           encountered. Note that the ``ipu_id`` is an index
                           in a multi-IPU device within PopTorch, and is
                           separate and distinct from the device ids used by
                           ``gc-info``.
        :param int phase_id: The PopART execution phase this code block should
                           belong on.
        :param layer_to_call: The layer to run on the specified IPU.
        """

        super().__init__()
        self.ipu_id = ipu_id
        self.phase_id = phase_id
        self.layer_to_call = layer_to_call

    def __call__(self, *input, **kwargs):
        begin_ipu_block(self.ipu_id, self.phase_id)
        out = self.layer_to_call(*input, **kwargs)
        return out


def custom_op(inputs, name, domain, domain_version, example_outputs):

    transformed_outputs = []
    for output in example_outputs:
        # Dead code which will get eliminated but will safely allow the same
        # input to be provided to example_output (since it is only supposed
        # to be a template). Otherwise the compiler may recognise th alias.
        transformed_outputs.append(torch.zeros_like(output))

    return torch.ops.poptorch.custom_operation(inputs, name, domain,
                                               domain_version,
                                               len(transformed_outputs),
                                               transformed_outputs)


def identity_loss(x, reduction="none"):
    """Marks this operation as being part of the loss calculation and, as such,
    will back-propagate through it in the PopTorch autograd. This enables
    multiple losses and custom losses.

    :param tensor loss: The calculated loss.
    :param string reduction: Reduce the loss output as per PyTorch loss
        semantics. Supported values are:
            * "none": Don't reduce
            * "sum": Sum the losses.
            * "mean": Take the mean of the losses.

    :returns: An identity loss custom op.
    """
    if reduction == "sum":
        return torch.ops.poptorch.identity_loss(x, 0)

    if reduction == "mean":
        return torch.ops.poptorch.identity_loss(x, 1)

    assert reduction == "none", "Unsupported reduction type!"
    return torch.ops.poptorch.identity_loss(x, 2)
