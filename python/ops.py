# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
from .logging import logger
from . import enums

begin_ipu_block = torch.ops.poptorch.begin_ipu_block
end_ipu_block = torch.ops.poptorch.end_ipu_block
ipu_print_tensor = torch.ops.poptorch.ipu_print_tensor
set_available_memory = torch.ops.poptorch.set_available_memory
nop = torch.ops.popart.nop


def apply_optimizer(optimizer):
    num_groups = len(optimizer.param_groups)
    for index in range(0, num_groups):
        torch.ops.poptorch.optimizer_group(
            index, optimizer.param_groups[index]["params"])


def serializedMatMul(lhs, rhs, mode, factor=0, keep_precision=False):
    """ Instantiate a matmul that should be serialized.

   The matrix multiplication will be split into separate smaller matmuls
   which will be executed in serie.

   :param lhs: Lhs input matrix
   :param rhs: Rhx input matrix
   :param poptorch.MatMulSerializationMode mode: Which dimension of the matmul
    to serialize on.
   :param int factor: Number of serialized matmuls. Must be a factor of the
   dimensions to serialize on.
   :param bool keep_precision: If True then any MatMul split along its
    reducing dimension will have an output type of float and a cast will be
    added after the addInplaces.
   """
    assert isinstance(keep_precision, bool)
    assert isinstance(factor, int)
    assert isinstance(mode, enums.MatMulSerializationMode)
    out = torch.matmul(lhs, rhs)
    return torch.ops.poptorch.set_matmul_serialization(out, mode.value, factor,
                                                       keep_precision)


def _assertIdIsValid(name, value, expected_type):
    assert isinstance(value, expected_type) or \
            (isinstance(value, int) and value >= 0), (
                f"{name} must be either a positive integer or a "
                f"{expected_type.__name__}")


class Block(torch.nn.Module):
    """Runs all layers called inside this scope on a specified IPU.


    >>> with poptorch.Block("IPU0"):
    ...     self.layer = MyLayer(x)

    """
    # Will be set by the ExecutionStrategy before the graph is traced.
    # If it's None then it means it's a CPU execution of the graph so
    # turn the whole class into a no-op.
    _stages_manager = None

    @staticmethod
    def useAutoId():
        """Call this method at the beginning of your forward() method to
        enable automatic Block Ids.

        Blocks with a None ``user_id`` will be assigned an automatic id
        which will be the index of this block in the list of id-less Blocks.

        >>> poptorch.Block.useAutoId()
        >>> with poptorch.Block(): # user_id = "0"
        ...     layer()
        >>> with poptorch.Block("special_block"): # user_id = "special_block"
        ...     layer()
        >>> with poptorch.Block(): # user_id = "1"
        ...     layer()
        """
        if Block._stages_manager is not None:
            Block._stages_manager.resetAutoId()

    def __init__(self, user_id=None, ipu_id=None):
        """

        :param user_id: Pipeline stage this code block should belong to.
                         All stages must have a unique, incrementing, id.
                         By default all layers will be in stage 0 until the
                         first pipeline annotation is encountered.
        :param int ipu_id: The id of the IPU to run on.
                       Note that the ``ipu_id`` is an index
                       in a multi-IPU device within PopTorch, and is
                       separate and distinct from the device ids used by
                       ``gc-info``.
        """
        super().__init__()
        self._user_id = user_id
        self._ipu_id = ipu_id

    def __enter__(self):
        if Block._stages_manager is not None:
            Block._stages_manager.beginStage(self._user_id, self._ipu_id)

    def __exit__(self, type, value, traceback):
        end_ipu_block()


class BeginBlock(torch.nn.Module):
    """Runs a layer on a specified Phase mapped to a specific API.

    All layers after this layer will also run on
    the same IPU until another IPU wrapper is encountered.

    By default this will be "pipelined" execution, however this
    can be overridden by the `poptorch.Options`.

    >>> self.layer = poptorch.BeginBlock(1, MyLayer(x))

    """

    def __init__(self, layer_to_call, user_id=None, ipu_id=None):
        """
        All subsequent layers of the network will be part of this phase until
        another layer is wrapped.

        :param int stage_id: Pipeline stage this code block should belong to.
                         All stages must have a unique, incrementing, id.
                         By default all layers will be in stage 0 until the
                         first pipeline annotation is encountered.
        :param layer_to_call: The layer to run on the specified IPU.
        :param phase_id: The PopART execution phase this code block should
                         belong on.
        :type phase_id: int >= 0 or poptorch.PhaseId.
        :param ipu_id: The id of the IPU to run on.
                       Note that the ``ipu_id`` is an index
                       in a multi-IPU device within PopTorch, and is
                       separate and distinct from the device ids used by
                       ``gc-info``.
        :type ipu_id: int >= 0 or poptorch.IpuId.
        """
        super().__init__()
        self._user_id = user_id
        self._layer_to_call = layer_to_call
        self._ipu_id = ipu_id

    def __call__(self, *input, **kwargs):
        if Block._stages_manager is not None:
            if self._user_id is None:
                self._user_id = Block._stages_manager.nextAutoId()
            Block._stages_manager.beginStage(self._user_id, self._ipu_id)

        out = self._layer_to_call(*input, **kwargs)
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

    :param torch.Tensor loss: The calculated loss.
    :param str reduction: Reduce the loss output as per PyTorch loss
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
