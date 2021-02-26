# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
from . import enums
from ._logging import logger

_end_ipu_block = torch.ops.poptorch.end_ipu_block


def ipu_print_tensor(tensor, title=""):
    return torch.ops.poptorch.ipu_print_tensor(tensor, title)


def for_loop(count, body, inputs):
    """ An on device for loop. This loop will execute on device for |count|
        number of iterations.

        The body should be a python function containing the PyTorch code you
        wish to execute in a loop. It should take as input the same number of
        tensors as it outputs. Each iteration will have the previous output
        passed in as input.

    :param count: Number of iterations of the loop.
    :param body: The function to be executed.
    :param inputs: The initial inputs to the functon.
    """

    if not isinstance(inputs, list):
        raise ValueError(("poptorch.for_loop expects input tensors (inputs)"
                          " to be a list of tensors. (Object is not list)"))

    for ind, tensor in enumerate(inputs):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                ("poptorch.for_loop expects input tensors (inputs) to be"
                 " a list of tensors. (Object contained in list at index"
                 " %d is not torch.tensor)") % ind)

    # Start the for loop.
    torch.ops.poptorch.start_for_loop(inputs)
    outputs = body(*inputs)

    # Break the alias of the outputs.
    example_outputs = []
    for output in outputs:
        example_outputs.append(torch.zeros(output.size()))

    if not isinstance(outputs, list) and not isinstance(outputs, tuple):
        outputs = [outputs]

    # End the for loop.
    return torch.ops.poptorch.end_for_loop(outputs, inputs, count,
                                           example_outputs)


def nop(tensor):
    """ A no-operation: it is functionally the same as an identity but is never
    elimated by PopART patterns or inlining, so it is useful for debugging.

    :param torch.Tensor tensor: the tensor to simply return by the no-op.
    :returns: The same tensor which was input.
    :rtype: torch.Tensor
    """
    return torch.ops.popart.nop(tensor)


def recomputationCheckpoint(*tensors):
    """Operation for checkpointing values in a computational pipeline stage.

    When recomputation is enabled, these values will not be recomputed and they
    will be stored in memory between forward and backwards passes instead.

    :param tensors: one or more tensors which should be checkpointed.
    :return: Tensors (same number and shape as the input tensors).
    :rtype: tuple
    """

    # Allow passing a single list or tuple
    if len(tensors) == 1:
        if isinstance(tensors[0], (tuple, list)):
            return type(tensors[0])(recomputationCheckpoint(*tensors[0]))

    out = []
    for t_in in tensors:
        if not isinstance(t_in, torch.Tensor):
            raise ValueError("All inputs must be tensors")

        out.append(torch.ops.poptorch.recomputation_checkpoint(t_in))

    if len(out) == 1:
        return out[0]

    # Return a tuple by default since Poptorch does not support list inputs
    return tuple(out)


def serializedMatMul(lhs, rhs, mode, factor=0, keep_precision=False):
    """ Calculates a matrix product using a serialized matrix multiplication.

    The matrix multiplication, lhs*rhs, is split into separate smaller
    multiplications, calculated one after the other, to reduce the memory
    requirements of the multiplication and its gradient calculation.

    :param torch.Tensor lhs: Left-hand size input matrix.
    :param torch.Tensor rhs: Right-hand side input matrix.
    :param poptorch.MatMulSerializationMode mode: Which dimension of the matmul
        to serialize on: for matrix A (m by n) multiplied by matrix B (n by p).
        * InputChannels: Split across the input channels (dimension m).
        * ReducingDim: Split aross the reducing dimension (n).
        * OutputChannels: Split across the output channels (dimenion p).
        * Disabled: Same as an ordinary matrix multiplication.
    :param int factor: Number of serialized multiplications. Must be a factor of
        the dimension to serialize on.
    :param bool keep_precision: (Half/float16 inputs only) The forward op when
        serializing over ReducingDim and the backwards ops when serializing over
        InputChannels involve an addition step. If ``keep_precision`` is True,
        these additions will occur using float32 rather than half precision
        partials, matching those used for the individual matrix multiplications.
   """
    assert isinstance(keep_precision, bool)
    assert isinstance(factor, int)
    assert isinstance(mode, enums.MatMulSerializationMode)
    out = torch.matmul(lhs, rhs)
    return torch.ops.poptorch.set_matmul_serialization(out, mode.value, factor,
                                                       keep_precision)


def set_available_memory(tensor, available_memory_proportion):
    """ Sets the available memory for a convolution or matrix multiplication.

    When called on the on the output of a convolution or a matrix
    multiplication, it sets the proportion of tile memory (between 0 and 1) to
    be made available as temporary memory for the convolution/matrix
    multipication. Less temporary memory will reduce the time performance but
    may use less memory overall. Lower memory proportions result in the use of
    more live (not tempoerary) memory, and so the overall memory may increase
    for too low values, possibly resulting in out of memory errors.

    In the event that the value is too low, the planner will replan for the
    smaller memory usage possible.

    >>> class BasicNetwork(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.conv = nn.Conv2d(4, 4, 3, stride=2)
    ...
    ...     def forward(self, x):
    ...         out = self.conv(x)
    ...         out = poptorch.set_available_memory(out, 0.2)
    ...         return out

    :param torch.Tensor tensor: output tensor of a convolution or matrix
        multiplication (otherwise the statement will be an identity).
    :param float available_memory_proportion: proportion between 0.0 and 1.0
        of tile memory to be made available for temporary memory (default 0.6).
    :returns: input tensor, as if calling an identity function.
    :rtype: torch.Tensor
     """
    return torch.ops.poptorch.set_available_memory(
        tensor, available_memory_proportion)


def _assertIdIsValid(name, value, expected_type):
    assert isinstance(value, expected_type) or \
            (isinstance(value, int) and value >= 0), (
                f"{name} must be either a positive integer or a "
                f"{expected_type.__name__}")


# The next two classes do not implement the forward method
# pylint: disable=abstract-method


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
        """Call this method at the beginning of your ``forward()`` method to
        enable automatic block id generation.

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

        :param user_id: A user defined identifier for the block.
            Blocks with the same id are considered as being a single block.
            Block identifiers are also used to manually specify pipelines or
            phases.
        :type user_id: str, optional
        :param int, optional ipu_id: The id of the IPU to run on.
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
        _end_ipu_block()


class BeginBlock(torch.nn.Module):
    """Runs all layers from the given layer until the beginning of the next
    block on a specified IPU.

    All layers after this layer will also run on
    the same IPU until another ``BeginBlock`` is encountered.

    By default :py:class:`PipelinedExecution` will be used, however this
    can be overridden in the `poptorch.Options`.

    .. seealso:: :py:meth:`poptorch.Options.setExecutionStrategy`

    >>> self.layer = poptorch.BeginBlock(MyLayer(x))

    """

    def __init__(self, layer_to_call, user_id=None, ipu_id=None):
        """
        All subsequent layers of the network will be part of this block until
        another layer is wrapped.

        :param torch.nn.Module layer_to_call: The layer to run on the
            specified IPU.
        :param user_id: A user defined identifier for the block.
            Blocks with the same id are considered as being a single block.
            Block identifiers are also used to manually create
            :py:class:`Stages<poptorch.Stage>` and
            :py:class:`Phases<poptorch.Phase>`.
        :type user_id: str, optional
        :param int, optional ipu_id: The id of the IPU to run on.
                       Note that the ``ipu_id`` is an index
                       in a multi-IPU device within PopTorch, and is
                       separate and distinct from the device ids used by
                       ``gc-info``.
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


# pylint: enable=abstract-method

# Store all attributes to prevent garbage collection
attributes_lists = []

ATTR_PREFIX = "attr:"


def custom_op(inputs,
              name,
              domain,
              domain_version,
              example_outputs,
              attributes=None):
    """Applies a custom operation, implemented within PopART, to the inputs.

    :param tuple inputs: A tuple of input tensors, for example, (x, y).
    :param str name: unique name of the PopART custom
    :param str domain: domain for the op
    :param int domain_version: version of the domain to use
    :param iterable example_outputs: a tuple of tensors with the same type
        and shape of the outputs; the value does not matter as all values will
        be set to zero for tracing purposes.
    :param dict attributes: a dictionary of attributes for the custom op. All
        attributes keys must be strings. All attribute values must be floats,
        ints, strings, or a list/tuple containing only floats, only ints or only
        strings (not a mix of types within the list).

    :returns: The outputs of the forward op of the custom op.
    """
    transformed_outputs = []
    for output in example_outputs:
        # Dead code which will get eliminated but will safely allow the same
        # input to be provided to example_output (since it is only supposed
        # to be a template). Otherwise the compiler may recognise the alias.
        transformed_outputs.append(torch.zeros_like(output))

    if attributes is not None:
        # Handle attributes list
        for k, v in attributes.items():
            if not isinstance(k, (str)):
                raise ValueError("All attribute keys must be strings.")
            if not isinstance(v, (float, int, str, list, tuple)):
                raise ValueError("Attribute values must be floats, ints, " +
                                 "strings or a list/tuple of float, ints of " +
                                 "strings.")

            if isinstance(v, (list, tuple)):
                for element in v:
                    if not isinstance(element, (type(v[0]))):
                        raise ValueError("The types in a list/tuple " +
                                         "attribute must all be the same.")

        # Non-ascii cannot be converted to std::string in C++
        def error_on_non_ascii(s):
            if isinstance(s, (list, tuple)):
                for v in s:
                    error_on_non_ascii(v)

            if not isinstance(s, str):
                return

            for ch in s:
                if ord(ch) >= 128:
                    raise ValueError(f"{s} contains non-ASCII characters.")

        for k in attributes.keys():
            error_on_non_ascii(k)

        for v in attributes.values():
            error_on_non_ascii(v)

        # The id should not change between traces, so we need to re-use any
        # attribute dictionaries. This more complicated because equality of
        # values is insufficient: [1, 2, 3] == [1.0, 2.0, 3.0]
        def same_attribute_types(candidate_att, search_attr):
            sorted_keys = sorted(candidate_att.keys())
            if sorted_keys != sorted(search_attr.keys()):
                return False

            for key in sorted_keys:
                candidate = candidate_att[key]
                search = search_attr[key]
                if not isinstance(candidate, (type(search))):
                    return False
                if isinstance(candidate, (list, tuple)):
                    if not isinstance(candidate[0], type(search[0])):
                        return False
            return True

        for attrib_cand in attributes_lists:
            if attrib_cand != attributes:
                continue

            # Equality does not imply same types
            if not same_attribute_types(attrib_cand, attributes):
                continue

            attributes = attrib_cand
            break
        else:
            attributes_lists.append(attributes)

    # NB None is a singleton in Python
    attributes_id_str = f"{ATTR_PREFIX}{hex(id(attributes))}"

    return torch.ops.poptorch.custom_operation(inputs, name, domain,
                                               domain_version,
                                               len(transformed_outputs),
                                               transformed_outputs,
                                               attributes_id_str)


def identity_loss(x, reduction):
    """Marks this operation as being part of the loss calculation and, as such,
    will back-propagate through it in the PopTorch autograd. This enables
    multiple losses and custom losses.

    :param torch.Tensor loss: The calculated loss.
    :param str reduction: Reduce the loss output as per PyTorch loss
        semantics. Supported values are:

        * ``"sum"``: Sum the losses.
        * ``"mean"``: Take the mean of the losses.
        * ``"none"``: Don't reduce the losses.

    :returns: An identity loss custom op.
    """
    if reduction == "sum":
        return torch.ops.poptorch.identity_loss(x, 0)

    if reduction == "mean":
        return torch.ops.poptorch.identity_loss(x, 1)

    assert reduction == "none", "Unsupported reduction type!"
    return torch.ops.poptorch.identity_loss(x, 2)


class MultiConv():
    """
    Combines all convolution layers evaluated inside this scope into a single
    multi-convolution.

    Multi-convolutions allow for a set of data-independent convolutions to be
    executed in parallel. Executing convolutions in parallel can lead to an
    increase in the data throughput.

    For example:

    >>> with poptorch.MultiConv():
    ...     y = self.convA(x)
    ...     v = self.convB(u)

    Combines the two data-independent convolutions into a single
    multi-convolution.

    Refer to the PopLibs documentation for further information on
    multi-convolutions.
    """

    def __init__(self):
        self._available_memory_proportions = None
        self._partials_types = None
        self._plan_type = None
        self._per_conv_reserved_tiles = None
        self._cycle_back_off = None

    @staticmethod
    def _validatePerConvProperty(name, value, expected_scalar_type):
        if value is None:
            return value

        if isinstance(value, expected_scalar_type):
            # Wrap as tuple
            return (value, )

        if isinstance(value, (list, tuple)) and len(value) > 0 and all(
                isinstance(x, expected_scalar_type) for x in value):
            return value

        raise AssertionError(f"Invalid {name}!")

    def availableMemoryProportions(self, value):
        """The available memory proportion per convolution, each [0, 1).

        :param value: Can be a ``float`` value in which case the same value is
            used for all of the convolutions. Otherwise, can be a ``tuple`` or
            ``list`` containing as many ``float`` values as the number of
            convolutions.
        :type value: float, [float]
        :returns: self, to support method chaining
        """
        name = "available memory proportion"
        value = self._validatePerConvProperty(name, value, float)
        self._available_memory_proportions = value
        return self

    def partialsTypes(self, value):
        """The partials type used for each convolution.

        :param value: Can be a single instance of ``torch.dtype`` in which case
            the same value is used for all of the convolutions. Otherwise, can
            be a ``tuple`` or ``list`` containing as many ``torch.dtype``
            values as the number of convolutions.
        :type value: :py:class:`torch.dtype`,
            [:py:class:`torch.dtype`]
        :returns: self, to support method chaining
        """

        # TODO(T34238): enums.MultiConvPartialsType deprecated in 2.0
        def encode_dtype(dtype):
            if dtype in [
                    torch.float, torch.float32,
                    enums.MultiConvPartialsType.Float
            ]:
                return enums.MultiConvPartialsType.Float.value
            if dtype in [
                    torch.half, torch.float16, enums.MultiConvPartialsType.Half
            ]:
                return enums.MultiConvPartialsType.Half.value
            raise ValueError(
                'Invalid partials types. Expecting torch.float or torch.half')

        if isinstance(value, (list, tuple)):
            value = [encode_dtype(v) for v in value]
            warn = any([not isinstance(v, torch.dtype) for v in value])
        else:
            value = (encode_dtype(value), )
            warn = not isinstance(value, torch.dtype)

        if warn:
            logger.warning('Usage of enum.MultiConvPartialsType is now '
                           'deprecated. Please use torch.float or '
                           'torch.half instead')

        self._partials_types = value
        return self

    def planType(self, value):
        """Select the multi-convolution execution strategy.

        :param value: An instance of :py:class:`MultiConvPlanType`.

        :returns: self, to support method chaining
        """
        if value is None:
            self._plan_type = value
        elif isinstance(value, enums.MultiConvPlanType):
            self._plan_type = value
        else:
            raise AssertionError("Invalid plan type!")

        return self

    def perConvReservedTiles(self, value):
        """Tiles to reserve for each convolution.

        :param value: Number of tiles
        :type value: int
        :returns: self, to support method chaining
        """
        assert isinstance(value, int)
        self._per_conv_reserved_tiles = value
        return self

    def cycleBackOff(self, value):
        """Cycle back off proportion.

        :param value: Number between 0 and 1
        :type value: float
        :returns: self, to support method chaining
        """
        assert isinstance(value, float)
        self._cycle_back_off = value
        return self

    def __enter__(self):
        torch.ops.poptorch.begin_multi_conv()

    def __exit__(self, type, value, traceback):
        # Convert enums to ints if set
        plan_type = self._plan_type
        if plan_type is not None:
            plan_type = plan_type.value

        torch.ops.poptorch.end_multi_conv(self._available_memory_proportions,
                                          self._partials_types, plan_type,
                                          self._per_conv_reserved_tiles,
                                          self._cycle_back_off)
