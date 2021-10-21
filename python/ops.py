# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from typing import Callable, Dict, List, Union, Tuple, Optional
import copy
import copyreg
import torch

from . import enums
from . import poptorch_core
from . import _impl

_end_ipu_block = torch.ops.poptorch.end_ipu_block


def ctc_beam_search_decoder(probs: "torch.Tensor",
                            lengths: "torch.Tensor",
                            blank: int = 0,
                            beam_width: int = 100,
                            top_paths: int = 1) -> List["torch.Tensor"]:
    """Add a connectionist temporal classification (CTC) beam search decoder
       to the model.

    Calculates the most likely top paths and their probabilities given the
    input logarithmic probabilities and the data lengths.

    :param probs: Logarithmic probabilities tensor with the shape
                               of [input_length, batch_size, num_classes].
    :param lengths: Tensor representing lengths of the inputs
                                 of shape [batch_size].
    :param blank: Integer identifier of the blank class (default: 0).
    :param beam_width: Number of beams used during decoding (default: 100).
    :param top_paths: Number of most likely paths to return (default: 1).
    :returns: Three tensors representing paths' probabilities - of shape
              [batch_size, top_paths], paths' lengths - of shape
              [batch_size, top_paths] and the decoded paths - of shape
              [batch_size, top_paths, input_length].
    """
    return torch.ops.poptorch.ctc_beam_search_decoder(probs, lengths, blank,
                                                      beam_width, top_paths)


def ipu_print_tensor(tensor: "torch.Tensor",
                     title: str = "") -> "torch.Tensor":
    """
    Adds an op to print the content of a given IPU tensor.

    When this is executed the tensor
    will be copied back to host and printed.

    When this operation is called in the backward pass it
    will print the gradient of the tensor.

    The operation is an identity operation and will return the exact same
    tensor. The returned tensor must be used in place of the original tensor
    in the rest of the program, to make sure that the print operation isn't
    optimised away.

    For example if the original code looks like this:

    .. code-block:: python

      def forward(self, c, d, b)
        a = c + d
        return a + b

    If the result of ``ipu_print_tensor`` is not used, it will be optimised
    out by the graph optimiser and tensor will not be printed.

    So if you want to print the value of `a`, you should do:

    .. code-block:: python

      def forward(self, c, d, b)
        a = c + d
        x = poptorch.ipu_print_tensor(a)
        return x + b

    Optionally, you may add a second string parameter to be used as a title.

    .. code-block:: python

      def forward(self, c, d, b)
          a = c + d
          x = poptorch.ipu_print_tensor(a, "summation"))
          return x + b


    .. warning::
       In order for the print operation to not be optimised out by the graph
       optimiser, you must use the output of the print.

    :param ipu_print_tensor: The tensor to print.
    :returns: The input unchanged.
    """
    return torch.ops.poptorch.ipu_print_tensor(tensor, title)


def for_loop(count: int,
             body: Callable[[List['torch.Tensor']], List['torch.Tensor']],
             inputs: List['torch.Tensor']) -> List['torch.Tensor']:
    """ An on device for loop. This loop will execute on device for `count`
        number of iterations.

        The body should be a python function containing the PyTorch code you
        wish to execute in a loop. It should take as input the same number of
        tensors as it outputs. Each iteration will have the previous output
        passed in as input.

    :param count: Number of iterations of the loop.
    :param body: The function to be executed.
    :param inputs: The initial inputs to the function.
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
        grad = output.requires_grad
        example_outputs.append(torch.zeros_like(output, requires_grad=grad))

    if not isinstance(outputs, list) and not isinstance(outputs, tuple):
        outputs = [outputs]

    # End the for loop.
    return torch.ops.poptorch.end_for_loop(outputs, inputs, count,
                                           example_outputs)


def nop(tensor: "torch.Tensor") -> "torch.Tensor":
    """A no-operation: it is functionally the same as an identity but is never
    eliminated by PopART patterns or inlining, so it is useful for
    debugging.

    :param tensor: the tensor to pass to the no-op.
    :returns: The same tensor which was input.
    """
    return torch.ops.popart.nop(tensor)


def recomputationCheckpoint(*tensors: List["torch.Tensor"]
                            ) -> List["torch.Tensor"]:
    """Operation for checkpointing values in a computational pipeline stage.

    When recomputation is enabled, these values will not be recomputed and they
    will be stored in memory between forward and backwards passes instead.

    :param tensors: One or more tensors which should be checkpointed.
    :return: Tensors (same number and shape as the input tensors).
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

    # Return a tuple by default since PopTorch does not support list inputs
    return tuple(out)


def serializedMatMul(lhs: "torch.Tensor",
                     rhs: "torch.Tensor",
                     mode: "poptorch.MatMulSerializationMode",
                     factor: int = 0,
                     keep_precision: bool = False) -> "torch.Tensor":
    """ Calculates a matrix product using a serialized matrix multiplication.

    The matrix multiplication, ``lhs*rhs``, is split into separate smaller
    multiplications, calculated one after the other, to reduce the memory
    requirements of the multiplication and its gradient calculation.

    :param lhs: Left-hand size input matrix.
    :param rhs: Right-hand side input matrix.
    :param mode: Which dimension of the matmul
        to serialize on: for matrix A (m by n) multiplied by matrix B (n by p).
        * InputChannels: Split across the input channels (dimension m).
        * ReducingDim: Split across the reducing dimension (n).
        * OutputChannels: Split across the output channels (dimension p).
        * Disabled: Same as an ordinary matrix multiplication.
    :param factor: Number of serialized multiplications. Must be a factor of
        the dimension to serialize on.
    :param keep_precision: (Half/float16 inputs only) The forward op when
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


def set_available_memory(tensor: "torch.Tensor",
                         available_memory_proportion: float) -> "torch.Tensor":
    """Sets the amount of temporary memory made available to an operation.

    The operators that can be tuned with this setting include:

    * convolution
    * matrix multiplication
    * embedding lookups
    * indexing operations

    When applied to the output of a supported operation, it controls the
    trade-off between execution cycles and the temporary memory used during the
    execution of the operation.

    The value should be between 0 and 1 (inclusive) and represents a proportion
    of available memory on the IPU. The default value is 0.6 (therefore, by
    default, PopTorch will not use more than 60% of IPU memory for temporary
    data).

    PopTorch passes this setting to the PopLibs operator planner, which will
    try to constrain the use of temporary memory to below this value. Generally,
    an operation that has more temporary memory available will run in fewer
    cycles.

    For a specific operation, the necessary amount of temporary memory may be
    more than amount specified by this option. In this case, a warning message
    will be generated.

    For more information, please refer to the `technical note
    <https://docs.graphcore.ai/projects/available-memory/en/latest/>`_ on
    optimising temporary memory usage.

    >>> class BasicNetwork(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.conv = nn.Conv2d(4, 4, 3, stride=2)
    ...
    ...     def forward(self, x):
    ...         out = self.conv(x)
    ...         out = poptorch.set_available_memory(out, 0.2)
    ...         return out

    :param tensor: Output tensor from a supported operation (otherwise the
        statement will be an identity).
    :param available_memory_proportion: Proportion between 0.0 and 1.0
        of tile memory to be made available for temporary memory (default 0.6).
    :returns: The input tensor, as if calling an identity function.
     """
    return torch.ops.poptorch.set_available_memory(
        tensor, available_memory_proportion)


def set_overlap_for_input(input_tensor: "torch.Tensor",
                          mode: "poptorch.OverlapMode") -> "torch.Tensor":
    """Sets host overlap setting for input_tensor.

    You can increase performance in some cases by overlapping the copying
    from the host to IPUs with computation. However, this requires a number
    of IPU tiles to be set aside as IO tiles using
    :py:func:`poptorch.options._TensorLocationOptions.numIOTiles` which may
    affect computation performance.

    You should use this function at the start of your model's `forward` method
    for each applicable input and use the returned tensor in future ops.

    :param input_tensor: The input tensor for which enable overlapping host IO.
    :param mode: Control to what extent the host IO overlaps computation.
    :returns: the input tensor, specified for overlap.

    .. seealso:: :py:class:`poptorch.OverlapMode`.
    """
    return torch.ops.poptorch.set_overlap_for_input(input_tensor, mode.value)


def set_overlap_for_output(output_tensor: "torch.Tensor",
                           mode: "poptorch.OverlapMode") -> "torch.Tensor":
    """Sets host overlap setting for output_tensor.

    You can increase performance in some cases by overlapping the copying
    from the IPUs to host with computation. However, this requires a number
    of IPU tiles to be set aside as IO tiles using
    :py:func:`poptorch.options._TensorLocationOptions.numIOTiles` which may
    affect computation performance.

    You should use this function at the end ofyour model's `forward` method
    for each applicable output just before returning the tensor.

    :param output_tensor: The output tensor for which enable overlapping host
      IO.
    :param mode: Control to what extent the host IO overlaps computation.
    :returns: the output tensor, specified for overlap.

    .. seealso:: :py:class:`poptorch.OverlapMode`.
    """
    return torch.ops.poptorch.set_overlap_for_output(output_tensor, mode.value)


def _assertIdIsValid(name, value, expected_type):
    assert isinstance(value, expected_type) or \
            (isinstance(value, int) and value >= 0), (
                f"{name} must be either a positive integer or a "
                f"{expected_type.__name__}")


# The next two classes do not implement the forward method
# pylint: disable=abstract-method


class Block(torch.nn.Module):
    """ A context manager to define blocks of the model.

    You can use ``Block`` as a context manager. This means you use Python's
    "with" statement as follows:

    >>> with poptorch.Block("Encoder"):
    ...     self.layer = MyLayer(x)

    All layers called inside this scope will run on the specified IPU, if
    one is specified. In addition, you can combine multiple blocks into
    a stage.

    .. seealso:: :py:meth:`poptorch.Options.setExecutionStrategy`

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

    def __init__(self,
                 user_id: Optional[str] = None,
                 ipu_id: Optional[int] = None):
        """

        :param user_id: A user defined identifier for the block.
            Blocks with the same id are considered as being a single block.
            Block identifiers are also used to manually specify pipelines or
            phases.
        :param ipu_id: The id of the IPU to run on.
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


# The pickle handler is needed for torch.save and for any copying.
# As the BlockModule class is defined within a function, it does not exist
# in a fresh python process. Hence, the easier way to recreate the object is to
# store the original model and call BeginBlock again
def _pickle_reduce_block(model):
    print(model.__class__)
    user_id = model.__dict__['_user_id']
    ipu_id = model.__dict__['_ipu_id']

    orig_model_class = model.__class__
    model.__class__ = model.__class__.__bases__[0]
    model_orig = copy.copy(model)
    model.__class__ = orig_model_class

    return BeginBlock, (model_orig, user_id, ipu_id)


# Used to allow BeginBlock to be used with a function
class LegacyBeginBlockFn(torch.nn.Module):
    def __init__(self, layer_to_call, user_id=None, ipu_id=None):
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


def BeginBlock(layer_to_call: torch.nn.Module,
               user_id: str = None,
               ipu_id: int = None) -> torch.nn.Module:
    """
    Define a block by modifying an existing PyTorch module.

    You can use this with an existing PyTorch module instance, as follows:

    >>> poptorch.BeginBlock(myModel.a_layer)
    >>> poptorch.BeginBlock(MyNewLayer())

    The wrapped module and all sub-modules will be part of this block until
    a sub-module is similar modified to be another block. In addition, if an IPU
    is specified, the module and its submodules will run on the specified IPU.

    You can combines multiple blocks into a stage.

    :param layer_to_call: PyTorch module to assign to the block.
    :param user_id: A user defined identifier for the block.
            Blocks with the same id are considered as being a single block.
            Block identifiers are also used to manually specify pipelines or
            phases.
    :param ipu_id: The id of the IPU to run on.
                   Note that the ``ipu_id`` is an index in a multi-IPU device
                   within PopTorch, and is separate and distinct from the device
                   ids used by ``gc-info``.

    .. seealso:: :py:meth:`poptorch.Options.setExecutionStrategy`
    """

    if not isinstance(layer_to_call, torch.nn.Module):
        # Previously, the function returned a new model so would work for any
        # callable. This was never documented but should still be permitted to
        # work.
        if callable(layer_to_call):
            return LegacyBeginBlockFn(layer_to_call, user_id, ipu_id)

        raise _impl.createPoptorchError(
            "module is not an instance of torch.nn.Module or " + "function.")

    class BlockModule(type(layer_to_call)):
        def __call__(self, *input, **kwargs):
            if Block._stages_manager is not None:
                if self._user_id is None:
                    self.__dict__['_user_id'] = (
                        Block._stages_manager.nextAutoId())
                Block._stages_manager.beginStage(self._user_id, self._ipu_id)

            return super().__call__(*input, **kwargs)

    if str(layer_to_call.__class__) == str(BlockModule):
        raise _impl.createPoptorchError(
            "module has already been assigned to a block.")

    BlockModule.__name__ = type(layer_to_call).__name__
    layer_to_call.__class__ = BlockModule
    layer_to_call.__dict__['_user_id'] = user_id
    layer_to_call.__dict__['_ipu_id'] = ipu_id

    # Register custom function to copy / serialize wrappers
    copyreg.pickle(BlockModule, _pickle_reduce_block)

    # There is no need to return as it is passed by reference, but this is for
    # backward compatibility
    return layer_to_call


# pylint: enable=abstract-method


def BlockFunction(user_id: Optional[str] = None, ipu_id: Optional[int] = None):
    """ A decorator to define blocks of the model.

    You can use ``BlockFunction`` as a decorator for an existing function, as
    follows:

    >>> @BlockFunction("Decoder", ipu_id=1)
    ... def decoder(self, encoder_output):
    ...     self.decoder_b1(encoder_output)

    All layers inside the function and any functions called by the function will
    run on the specified IPU, if one is specified. In addition, you can combine
    multiple blocks into a stage.

    :param user_id: A user defined identifier for the block.
        Blocks with the same id are considered as being a single block.
        Block identifiers are also used to manually specify pipelines or
        phases.
    :param ipu_id: The id of the IPU to run on.
                   Note that the ``ipu_id`` is an index
                   in a multi-IPU device within PopTorch, and is
                   separate and distinct from the device ids used by
                   ``gc-info``.

    .. seealso:: :py:meth:`poptorch.Options.setExecutionStrategy`
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with Block(user_id, ipu_id):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Store all attributes to prevent garbage collection
attributes_lists: List[Dict[str, Union[float, int, str, list, tuple]]] = []

ATTR_PREFIX = "attr:"


def custom_op(inputs: Tuple["torch.Tensor"],
              name: str,
              domain: str,
              domain_version: int,
              example_outputs: Tuple["torch.Tensor"],
              attributes: Optional[
                  Dict[str, Union[float, int, str, list, tuple]]] = None
              ) -> List["torch.Tensor"]:
    """Applies a custom operation, implemented within PopART, to the inputs.

    :param tuple inputs: A tuple of input tensors, for example, (x, y).
    :param str name: Unique name of the PopART custom op.
    :param str domain: Domain for the op.
    :param int domain_version: Version of the domain to use.
    :param iterable example_outputs: A tuple of tensors with the same type
        and shape as the outputs. The value does not matter as all values will
        be set to zero for tracing purposes.
    :param dict attributes: A dictionary of attributes for the custom op. All
        attribute keys must be strings. All attribute values must be floats,
        ints, strings, or a list/tuple containing only floats, only ints or only
        strings (not a mix of types within the list).

    :returns: The outputs of the forward op of the custom op.
    """
    transformed_outputs = []
    for output in example_outputs:
        # Dead code which will get eliminated but will safely allow the same
        # input to be provided to example_output (since it is only supposed
        # to be a template). Otherwise the compiler may recognise the alias.
        grad = output.requires_grad
        transformed_outputs.append(torch.zeros_like(output,
                                                    requires_grad=grad))

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


class CPU:
    def __init__(self, layer_to_call, ID):
        self._layer_to_call = layer_to_call

        self._ID = ID
        self.inputs = None
        self.outputs = None

    def execute(self):
        outs = self._layer_to_call(*self.inputs)

        if isinstance(outs, (list, tuple)):
            for persistent_output, output in zip(self.outputs, outs):
                persistent_output.copy_(output)
        else:
            self.outputs[0].copy_(outs)

    def createPersistentData(self, input, outputs):
        self.inputs = input
        self.outputs = [output.clone().contiguous() for output in outputs]

    def registerPersistentData(self):
        poptorch_core.registerBuffersWithCallback(self._ID, self.inputs,
                                                  self.outputs)

    def __call__(self, *input, **kwargs):
        # Mark all subsquent ops as happening on the host.
        torch.ops.poptorch.call_cpu_op([*input], self._ID)

        # Keep the trace happy by actually calling the layer.
        outputs = self._layer_to_call(*input)

        # Did we originally just output a single tensor?
        originally_single_tensor = False

        # Slight fixup for single tensor outputs.
        if not isinstance(outputs, (list, tuple)):
            originally_single_tensor = True
            outputs = [outputs]

        # Move the outputs and inputs into a permanent buffer.
        self.createPersistentData(input, outputs)

        # End CPU host execution and show the JIT what the output looks like.
        outputs = torch.ops.poptorch.end_cpu_op(outputs)

        # Register this callback with poptorch so it knows what to call.
        poptorch_core.registerCPUCallBack(self, self._ID)

        # Just return one tensor if it was supposed to be just one.
        if originally_single_tensor:
            return outputs[0]

        return outputs


def identity_loss(x: "torch.Tensor", reduction: "str") -> "torch.Tensor":
    """Marks this operation as being part of the loss calculation and, as such,
    will back-propagate through it in the PopTorch autograd. This enables
    multiple losses and custom losses.

    :param x: The calculated loss.
    :param reduction: Reduce the loss output as per PyTorch loss
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
        self._enable_conv_ditherings = None

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

    def availableMemoryProportions(self, value: Union[float, List[float]]
                                   ) -> "poptorch.MultiConv":
        """The available memory proportion per convolution, each [0, 1).

        For more information, please refer to the `technical note
        <https://docs.graphcore.ai/projects/available-memory/en/latest/>`_ on
        optimising temporary memory usage.

        :param value: Can be a ``float`` value in which case the same value is
            used for all of the convolutions. Otherwise, can be a ``tuple`` or
            ``list`` containing as many ``float`` values as the number of
            convolutions.
        :returns: self, to support method chaining
        """
        name = "available memory proportion"
        value = self._validatePerConvProperty(name, value, float)
        self._available_memory_proportions = value
        return self

    def partialsTypes(self, value: Union[torch.dtype, List[torch.dtype]]
                      ) -> "poptorch.MultiConv":
        """The partials type used for each convolution.

        :param value: Can be a single instance of ``torch.dtype`` in which case
            the same value is used for all of the convolutions. Otherwise, can
            be a ``tuple`` or ``list`` containing as many ``torch.dtype``
            values as the number of convolutions.
        :returns: self, to support method chaining
        """

        def encode_dtype(dtype):
            if dtype in [torch.float, torch.float32]:
                return 0
            if dtype in [torch.half, torch.float16]:
                return 1
            raise ValueError(
                'Invalid partials types. Expecting torch.float or torch.half')

        if isinstance(value, (list, tuple)):
            value = [encode_dtype(v) for v in value]
        else:
            value = (encode_dtype(value), )

        self._partials_types = value
        return self

    def enableConvDithering(self, value: Union[bool, List[bool]]
                            ) -> "poptorch.MultiConv":
        """Enable per-convolution dithering.

        :param value: Can be a ``bool`` value in which case the same value is
            used for all of the convolutions. Otherwise, can be a ``tuple`` or
            ``list`` containing as many ``bool`` values as the number of
            convolutions.
        :returns: self, to support method chaining
        """

        if value is None:
            self._enable_conv_ditherings = value
        elif isinstance(value, (list, tuple)):
            for x in value:
                if not isinstance(x, bool):
                    raise ValueError("value must be bool or list of bools")
            self._enable_conv_ditherings = value
        elif isinstance(value, bool):
            self._enable_conv_ditherings = (value, )
        else:
            raise ValueError("value must be bool or list of bools")
        return self

    def planType(self,
                 value: "poptorch.MultiConvPlanType") -> "poptorch.MultiConv":
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

    def perConvReservedTiles(self, value: int) -> "poptorch.MultiConv":
        """Tiles to reserve for each convolution.

        :param value: Number of tiles
        :returns: self, to support method chaining
        """
        assert isinstance(value, int)
        self._per_conv_reserved_tiles = value
        return self

    def cycleBackOff(self, value: float) -> "poptorch.MultiConv":
        """Cycle back off proportion.

        :param value: Number between 0 and 1
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
                                          self._cycle_back_off,
                                          self._enable_conv_ditherings)


class NameScope:
    """ Create a name scope for a code block. All operators originating
        from this block will have their names prefixed by the given string.

        >>> with poptorch.NameScope("CustomString"):
        ...     y = self.bmm(a, b)
        ...     z = torch.relu(y)
    """

    def __init__(self, name: str):
        assert isinstance(name, str), 'Parameter to NameScope must be a string'
        self.name = name

    def __enter__(self):
        torch.ops.poptorch.push_name_scope(self.name)

    def __exit__(self, type, value, traceback):
        torch.ops.poptorch.pop_name_scope()
