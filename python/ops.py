# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from collections import OrderedDict
from typing import Callable, Dict, List, Union, Tuple, Optional
import torch

from . import enums
from . import poptorch_core
from . import _impl
from ._utils import ATTR_PREFIX, flattenTensorStructure, reconstructTensorStructure

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
    if not isinstance(probs, torch.Tensor):
        raise _impl.createPoptorchError(
            "ctc_beam_search_decoder: probs must be a torch.tensor argument. "
            f"{type(probs)} is not supported.")
    if not isinstance(lengths, torch.Tensor):
        raise _impl.createPoptorchError(
            "ctc_beam_search_decoder: lengths must be a torch.tensor argument. "
            f"{type(lengths)} is not supported.")
    return torch.ops.poptorch.ctc_beam_search_decoder(probs, lengths, blank,
                                                      beam_width, top_paths)


def ipu_print_tensor(tensor: "torch.Tensor",
                     title: str = "",
                     print_gradient: bool = True,
                     summarise_threshold: int = 1000,
                     edge_items: int = 3,
                     max_line_width: int = 80,
                     digits: int = 4,
                     float_format: str = "auto",
                     separator: str = ", ",
                     open_bracket: str = "(",
                     close_bracket: str = ")") -> "torch.Tensor":
    """Adds an op to print the contents of the IPU tensor.

    When this is executed the tensor
    will be copied back to host and printed.

    When this operation is called in the backward pass it
    will print the gradient of the tensor.

    The operation is an identity operation and will return the exact same
    tensor. The returned tensor must be used in place of the original tensor
    in the rest of the program, to make sure that the print operation isn't
    optimised away.

    For example, if the original code looks like this:

    .. code-block:: python

      def forward(self, c, d, b)
        a = c + d
        return a + b

    If the result of ``ipu_print_tensor()`` is not used, the function will be
    optimised out by the graph optimiser and the tensor will not be printed.

    So if you want to print the value of `a`, you should do:

    .. code-block:: python

      def forward(self, c, d, b)
        a = c + d
        x = poptorch.ipu_print_tensor(a)
        return x + b

    Optionally, you can add a second string argument to be used as a title, as
    shown in the following example.
    The value of `a` will be printed after the title "summation". The value of
    the gradient of `a` will be printed after the title "summation_gradient" if
    the operation is called in the backward pass.

    .. code-block:: python

      def forward(self, c, d, b)
          a = c + d
          x = poptorch.ipu_print_tensor(a, "summation"))
          return x + b


    .. warning::
       To prevent the print operation being optimised out by the graph
       optimiser, you must use the output of the print.

    :param tensor: The tensor to print.
    :param title: An optional title to print before the tensor value.
        Defaults to "".
    :param print_gradient: Whether to print the gradient tensor associated
        with this tensor. Defaults to True.
    :param summarise_threshold: If the number of elements of the
        tensor exceeds this threshold the output will be summarised. Only the
        edge elements will be displayed with an ellipsis indicating skipped
        elements. A value of 0 will disable summarisation. Defaults to 1000.
    :param edge_items: Number of edge elements to include at the
        beginning and end when summarisation is enabled. Defaults to 3.
    :param max_line_width: Lines longer than this limit will be split
        across multiple lines. A value of 0 will disable line splitting.
        Defaults to 75.
    :param digits: Number of digits to display. For integers this limit can be
        exceeded if any number is large enough. For floating points this does
        not include the exponent. The number of digits is used in conjunction
        analysis of the tensor to determine the width of each element to align
        all elements when printed. A value of 0 disables this analysis and each
        elements will be printed in an unaligned format. Defaults to 4.
    :param float_format: Determines the floating point format to use. Automatic
        mode determines the appropriate format based on the data.
        Defaults to "auto".
        One of:
        - "auto": Automatically determine the format through analysis.
        - "fixed": Use fixed point e.g. -100.00.
        - "scientific": Use scientific notation e.g. -1.123e+10.
        - "none": Do not display all elements with the same format
    :param separator: Character used to delineate values. Defaults to " ".
    :param open_bracket: Character used to open a tensor. Defaults to "[".
    :param close_bracket: Character used to close a tensor. Defaults to "]".
    :returns: The input tensor unchanged.
    """
    if not isinstance(tensor, torch.Tensor):
        raise _impl.createPoptorchError(
            "ipu print tensor must take a torch.tensor argument. "
            f"{type(tensor)} is not supported.")
    float_format_dict = {"auto": 0, "fixed": 1, "scientific": 2, "none": 3}
    return torch.ops.poptorch.ipu_print_tensor(tensor, title,
                                               int(print_gradient),
                                               summarise_threshold, edge_items,
                                               max_line_width, digits,
                                               float_format_dict[float_format],
                                               separator, open_bracket,
                                               close_bracket)


def for_loop(count: int,
             body: Callable[[List['torch.Tensor']], List['torch.Tensor']],
             inputs: List['torch.Tensor']) -> List['torch.Tensor']:
    """An on-device for loop. This loop will execute on device for `count`
    number of iterations.

    The body should be a Python function containing the PyTorch code you
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

    # Clone the inputs to make sure ir reflects the fact that
    # body inputs are passed by value rather than by reference.
    cloned_inputs = [t.clone() for t in inputs]

    # Start the for loop.
    torch.ops.poptorch.start_for_loop(cloned_inputs)
    outputs = body(*cloned_inputs)
    if not isinstance(outputs, list) and not isinstance(outputs, tuple):
        outputs = [outputs]

    # End the for loop.
    res = torch.ops.poptorch.end_for_loop(outputs, cloned_inputs, count)

    return res


def cond(condition: 'torch.Tensor',
         then_body: Callable[[List['torch.Tensor']], List['torch.Tensor']],
         then_inps: List['torch.Tensor'],
         else_body: Callable[[List['torch.Tensor']], List['torch.Tensor']],
         else_inps: List['torch.Tensor']) -> List['torch.Tensor']:
    """An on-device if/else operation. This creates two branches of instructions
    executed conditionally on the device. Only for inference.

    The `then_body` and `else_body` should be Python functions containing the
    PyTorch code you wish to execute conditionally on the device. The condition
    is passed in the form of a boolean `Tensor` and the branch to be executed is
    decided in runtime directly on the device. There are a few conditions on the
    branch functions:

    * `then_body` and `else_body` can accept an arbitrary number of inputs
      (including zero).
    * Tensors defined in the `cond` caller (the outer graph) can be used inside
      `then_body` and `else_body` implicitly just as if they were passed
      through the inputs list.
    * `then_body` and `else_body` have to return the same number of
      corresponding outputs. This is because the result of the `cond` op is
      assigned to a common list of tensors.
    * all the tensors utilized by `then_body` and `else_body` are passed in by
      copy, so updating any of the tensors inside `then_body` and `else_body`
      does not affect the original tensors. To update a tensor passed in, its
      new value has to be returned from the body and assigned to the original
      tensor (please note that the number of outputs from `then_body` and
      `else_body` has to match).

    :param condition: The condition controlling the execution of `then_body` and
        `else_body`.
    :param then_body: The function to be executed if `condition` is True.
    :param then_inps: `then_body` input tensors.
    :param else_body: The function to be executed if `condition` is False.
    :param else_inps: `else_body` input tensors.
    """

    if not isinstance(then_inps, list) or not isinstance(else_inps, list):
        raise ValueError(
            ("poptorch.cond expects then_inps and else_inps tensors"
             " to be a list of tensors. (Object is not list)"))

    if not _impl.isRunningOnIpu():
        # CPU execution path
        if condition:
            res = then_body(*then_inps)
            return [res] if isinstance(res, torch.Tensor) else [*res]
        res = else_body(*else_inps)
        return [res] if isinstance(res, torch.Tensor) else [*res]

    # Clone the inputs to make sure ir reflects the fact that
    # body inputs are passed by value rather than by reference.
    cloned_condition = condition.clone()

    # Start the if block.
    torch.ops.poptorch.start_if_block(cloned_condition)

    outputs_then = then_body(*then_inps)
    if not isinstance(outputs_then, list) and not isinstance(
            outputs_then, tuple):
        outputs_then = [outputs_then]

    # Start the else block.
    torch.ops.poptorch.start_else_block(outputs_then)

    outputs_else = else_body(*else_inps)
    if not isinstance(outputs_else, list) and not isinstance(
            outputs_else, tuple):
        outputs_else = [outputs_else]

    return torch.ops.poptorch.end_if_block(outputs_else, cloned_condition)


def nop(tensor: "torch.Tensor") -> "torch.Tensor":
    """A no-operation: it is functionally the same as an identity but is never
    eliminated by PopART patterns or inlining, so it is useful for
    debugging.

    :param tensor: The tensor to pass to the no-op.
    :returns: The same tensor which was input.
    """
    if not isinstance(tensor, torch.Tensor):
        raise _impl.createPoptorchError(
            f"nop must take a torch.tensor argument. {type(tensor)} is not "
            "supported.")
    return torch.ops.poptorch.nop(tensor)


def dynamic_slice(tensor: "torch.Tensor", dim: int, start: "torch.Tensor",
                  size: int, step: int) -> "torch.Tensor":
    """Torch native dynamic slices can't be properly intercepted by backends,
    so this op is provided to enable dynamic slicing in poptorch applications.

    :param tensor: The tensor to slice.
    :param dim: The dimension to slice along.
    :param start: The start index.
    :param size: The slice size. Must be a constant int.
    :param step: The slice step. Must be a constant int.
    :returns: The sliced tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise _impl.createPoptorchError(
            f"dynamic_slice must take a torch.tensor input. {type(tensor)} is "
            "not supported.")
    if not isinstance(dim, int):
        raise _impl.createPoptorchError("Dimension must be an integer.")
    if not isinstance(start, torch.Tensor):
        raise _impl.createPoptorchError(
            "Slice start argument to dynamic_slice must be a torch.tensor. "
            f"{type(tensor)} is not supported.")
    if not isinstance(size, int):
        raise _impl.createPoptorchError("Size must be an integer.")
    if not isinstance(step, int):
        raise _impl.createPoptorchError("Step must be an integer.")
    return torch.ops.poptorch.dynamic_slice(tensor, dim, start, size, step)


def dynamic_update(input: "torch.Tensor", src: "torch.Tensor", dim: int,
                   start: "torch.Tensor", size: int) -> "torch.Tensor":
    """Torch native dynamic slices can't be properly intercepted by backends,
    so this op is provided to enable dynamic update slice in poptorch
    applications.

    :param input: The tensor to update.
    :param src: The tensor to embed into `input`
    :param dim: The dimension to slice along.
    :param start: The start index.
    :param size: The slice size. Must be a constant int.
    :returns: The sliced tensor.
    """
    if not isinstance(input, torch.Tensor):
        raise _impl.createPoptorchError(
            f"dynamic_update must take a torch.tensor input. {type(input)} is "
            "not supported.")
    if not isinstance(dim, int):
        raise _impl.createPoptorchError("Dimension must be an integer.")
    if not isinstance(start, torch.Tensor):
        raise _impl.createPoptorchError(
            "Slice start argument to dynamic_update must be a torch.tensor. "
            f"{type(start)} is not supported.")
    if not isinstance(src, torch.Tensor):
        raise _impl.createPoptorchError(
            "Src argument to dynamic_update must be a torch.tensor. "
            f"{type(src)} is not supported.")
    if not isinstance(size, int):
        raise _impl.createPoptorchError("Size must be an integer.")
    if input.dim() != src.dim():
        raise _impl.createPoptorchError(
            "input and src tensors must have same dimensionality. "
            f"({input.dim()}) vs ({src.dim()})")
    if input.dtype != src.dtype:
        raise _impl.createPoptorchError(
            "input and src tensor must have same dtype. "
            f"({input.dtype} vs {src.dtype})")
    return torch.ops.poptorch.dynamic_update(input, src, dim, start, size)


def recomputationCheckpoint(*tensors: List["torch.Tensor"]
                            ) -> List["torch.Tensor"]:
    """Operation for checkpointing values in a computational pipeline stage.

    When recomputation is enabled, these values will not be recomputed and they
    will be stored in memory between forward and backwards passes instead.

    :param tensors: One or more tensors which should be check-pointed.
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

    :param lhs: Left-hand side input matrix.
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
    if not isinstance(tensor, torch.Tensor):
        raise _impl.createPoptorchError(
            "You may only set available memory for torch.tensor values. "
            f"{type(tensor)} is not supported.")
    return torch.ops.poptorch.set_available_memory(
        tensor, available_memory_proportion)


def set_overlap_for_input(input_tensors, mode: "poptorch.OverlapMode"):
    """Sets host overlap setting for input_tensors.

    You can increase performance in some cases by overlapping the copying
    from the host to IPUs with computation. However, this requires a number
    of IPU tiles to be set aside as IO tiles using
    :py:func:`~poptorch.options._TensorLocationOptions.numIOTiles` which may
    affect computation performance.

    You should use this function at the start of your model's `forward` method
    for each applicable input and use the returned tensors in future ops.

    :param input_tensors: The input tensors for which enable overlapping host
      IO. This can be either a single tensor, or any combination of tuple,
      list, or dict of tensors.
    :param mode: Control to what extent the host IO overlaps computation.
    :returns: the input tensors, specified for overlap.

    .. seealso:: :py:class:`~poptorch.OverlapMode`.
    """

    def set_overlap_for_input_tensor(tensor):
        if not isinstance(tensor, torch.Tensor):
            raise _impl.createPoptorchError(
                "You may only set overlap for torch.tensor inputs. "
                f"{type(tensor)} is not supported.")
        return torch.ops.poptorch.set_overlap_for_input(tensor, mode.value)

    flattened = flattenTensorStructure(input_tensors)
    return reconstructTensorStructure(
        input_tensors, map(set_overlap_for_input_tensor, flattened))


def set_overlap_for_output(output_tensors, mode: "poptorch.OverlapMode"):
    """Sets host overlap setting for output_tensors.

    You can increase performance in some cases by overlapping the copying
    from the IPUs to host with computation. However, this requires a number
    of IPU tiles to be set aside as IO tiles using
    :py:func:`~poptorch.options._TensorLocationOptions.numIOTiles` which may
    affect computation performance.

    You should use this function at the end of your model's `forward` method,
    for each applicable output, just before returning the tensors.

    :param output_tensors: The output tensors to enable overlapping host
      IO for. This can be either a single tensor, or any combination of tuple,
      list, or dict of tensors.
    :param mode: Control to what extent the host IO overlaps computation.
    :returns: the output tensors, specified for overlap.

    .. seealso:: :py:class:`~poptorch.OverlapMode`.
    """

    def set_overlap_for_output_tensor(tensor):
        if not isinstance(tensor, torch.Tensor):
            raise _impl.createPoptorchError(
                "You may only set overlap for torch.tensor outputs. "
                f"{type(tensor)} is not supported.")
        return torch.ops.poptorch.set_overlap_for_output(tensor, mode.value)

    flattened = flattenTensorStructure(output_tensors)
    return reconstructTensorStructure(
        output_tensors, map(set_overlap_for_output_tensor, flattened))


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

    .. seealso:: :py:meth:`~poptorch.Options.setExecutionStrategy`

    """
    # Will be set by the ExecutionStrategy before the graph is traced.
    # If it's None then it means it's a CPU execution of the graph so
    # turn the whole class into a no-op.
    _stages_manager = None

    @staticmethod
    def useAutoId():
        """Call this method at the beginning of your ``forward()`` method to
        enable automatic block ID generation.

        Blocks with a None ``user_id`` will be assigned an automatic ID
        which will be the index of this block in the list of ID-less Blocks.

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

    @staticmethod
    def start(user_id: Optional[str] = None, ipu_id: Optional[int] = None):
        if Block._stages_manager is not None:
            Block._stages_manager.beginStage(user_id, ipu_id)

    def __init__(self,
                 user_id: Optional[str] = None,
                 ipu_id: Optional[int] = None):
        """

        :param user_id: A user defined identifier for the block.
            Blocks with the same ID are considered as being a single block.
            Block identifiers are also used to manually specify pipelines or
            phases.
        :param ipu_id: The ID of the IPU to run on.
                       Note that the ``ipu_id`` is an index
                       in a multi-IPU device within PopTorch, and is
                       separate and distinct from the device ids used by
                       ``gc-info``.
        """
        super().__init__()
        self._user_id = user_id
        self._ipu_id = ipu_id

    def __enter__(self):
        Block.start(self._user_id, self._ipu_id)

    def __exit__(self, type, value, traceback):
        _end_ipu_block()


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


class _BlockHook():
    """ A hook to define the blocks of the model.

    You can use ``_BlockHook`` as a forward_pre_hook for a ``torch.nn.Module``
    as follows:
    >>> m.register_forward_pre_hook(_BlockHook(user_id, ipu_id))

    All layers called after the hook has run will be run on the specified IPU,
    if one is specified. In addition, you can combine multiple blocks into a
    stage.

    .. seealso:: :py:meth:`~poptorch.Options.setExecutionStrategy`
    """

    def __init__(self, user_id, ipu_id) -> None:
        super().__init__()
        self._user_id = user_id
        self._ipu_id = ipu_id

    def __call__(self, module, input):
        if Block._stages_manager is not None:
            if self._user_id is None:
                self._user_id = (Block._stages_manager.nextAutoId())
            Block._stages_manager.beginStage(self._user_id, self._ipu_id)

    def __repr__(self):
        return f"BeginBlock(user_id={self._user_id}, ipu_id={self._ipu_id})"


def removeBlocks(module):
    """Recursively remove BeginBlock annotations from a Module if it
    contains any.

    :param torch.nn.Module module: Module to recursively unwrap.
    """
    assert isinstance(module, torch.nn.Module)
    for m in module.modules():
        # pylint: disable=protected-access
        m._forward_pre_hooks = OrderedDict(
            filter(lambda elt: not isinstance(elt[1], _BlockHook),
                   m._forward_pre_hooks.items()))


def BeginBlock(layer_to_call: torch.nn.Module,
               user_id: str = None,
               ipu_id: int = None) -> torch.nn.Module:
    """
    Define a block by modifying an existing PyTorch module.

    You can use this with an existing PyTorch module instance, as follows:

    >>> poptorch.BeginBlock(myModel.a_layer)
    >>> poptorch.BeginBlock(MyNewLayer())

    The module and all sub-modules will be part of this block until a
    sub-module is modified to be in another block. In addition, if an IPU is
    specified, the module and its submodules will run on the specified IPU.

    You can combine multiple blocks into a stage.

    :param layer_to_call: PyTorch module to assign to the block.
    :param user_id: A user defined identifier for the block.
            Blocks with the same ID are considered as being a single block.
            Block identifiers are also used to manually specify pipelines or
            phases.
    :param ipu_id: The ID of the IPU to run on.
                   Note that the ``ipu_id`` is an index in a multi-IPU device
                   within PopTorch, and is separate and distinct from the device
                   IDs used by ``gc-info``.

    .. seealso:: :py:meth:`~poptorch.Options.setExecutionStrategy`
    """

    if not isinstance(layer_to_call, torch.nn.Module):
        # Previously, the function returned a new model so would work for any
        # callable. This was never documented but should still be permitted to
        # work.
        if callable(layer_to_call):
            return LegacyBeginBlockFn(layer_to_call, user_id, ipu_id)

        raise _impl.createPoptorchError(
            "module is not an instance of torch.nn.Module or " + "function.")

    # pylint: disable=protected-access
    if any(
            isinstance(hook, _BlockHook)
            for hook in layer_to_call._forward_pre_hooks.values()):
        raise _impl.createPoptorchError(
            "module has already been assigned to a block.")

    layer_to_call.register_forward_pre_hook(_BlockHook(user_id, ipu_id))

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
        Blocks with the same ID are considered as being a single block.
        Block identifiers are also used to manually specify pipelines or
        phases.
    :param ipu_id: The ID of the IPU to run on.
                   Note that the ``ipu_id`` is an index
                   in a multi-IPU device within PopTorch, and is
                   separate and distinct from the device IDs used by
                   ``gc-info``.

    .. seealso:: :py:meth:`~poptorch.Options.setExecutionStrategy`
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with Block(user_id, ipu_id):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Store all attributes to prevent garbage collection
attributes_lists: List[Dict[str, Union[float, int, str, list, tuple]]] = []


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
        transformed_outputs.append(
            torch.zeros_like(output, requires_grad=grad, device=output.device))

    if attributes is not None:
        # Handle attributes list
        for k, v in attributes.items():
            if not isinstance(k, (str)):
                raise ValueError("All attribute keys must be strings.")
            if not isinstance(v, (float, int, str, list, tuple)):
                raise ValueError("Attribute values must be floats, ints, "
                                 "strings or a list/tuple of float, ints of "
                                 "strings.")

            if isinstance(v, (list, tuple)):
                for element in v:
                    if not isinstance(element, (type(v[0]))):
                        raise ValueError("The types in a list/tuple "
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
    """Allow the execution of a CPU op in the middle of an inference IPU graph.

    .. important:: CPU ops are only supported in inference graphs.

    Example:

    >>> class Model(torch.nn.Module):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>         self.cpu = poptorch.CPU(self.myCpuOp, "MyCPUOp")
    >>>
    >>>     def myCpuOp(self, x):
    >>>         return x * 2.0
    >>>
    >>>     def forward(self, x):
    >>>         # The arguments passed to "cpu" are forwarded to "myCpuOp"
    >>>         out = self.cpu(x)
    >>>         out = self.cpu(out)
    >>>         out = self.cpu(out)
    >>>         return out
    """

    def __init__(self, layer_to_call: Callable, ID: str):
        """
        Execute a given function on the CPU.

        :param: layer_to_call Python function to execute on the CPU. The
                              arguments passed when the CPU wrapper is called
                              will be forwarded to layer_to_call.
        :param: ID            Name of the CPU op.
        """
        self._layer_to_call = layer_to_call

        if isinstance(self._layer_to_call, torch.nn.Module):
            self._layer_to_call.requires_grad_(False)

        self._ID = ID

        self.in_shapes = None
        self.out_shapes = None

        self.inputs = None
        self.outputs = None

    def execute(self):
        """Implementation detail."""
        outs = self._layer_to_call(*self.inputs)

        if isinstance(outs, (list, tuple)):
            for persistent_output, output in zip(self.outputs, outs):
                persistent_output.copy_(output)
        else:
            self.outputs[0].copy_(outs)

    def registerPersistentData(self):
        """Implementation detail."""

        self.inputs = [torch.zeros(i, device='cpu') for i in self.in_shapes]
        self.outputs = [torch.zeros(o, device='cpu') for o in self.out_shapes]

        poptorch_core.registerBuffersWithCallback(self._ID, self.inputs,
                                                  self.outputs)

    def __call__(self, *input, **kwargs):
        """Implementation detail."""
        # Mark all subsquent ops as happening on the host.
        torch.ops.poptorch.call_cpu_op([*input], self._ID)

        if _impl.isRunningOnIpu():
            cpu_input = [
                torch.zeros_like(i,
                                 device="cpu",
                                 requires_grad=i.requires_grad) for i in input
            ]
        else:
            cpu_input = input

        # Keep the trace happy & get output shapes by actually calling the
        # layer.
        cpu_outputs = self._layer_to_call(*cpu_input)

        # Did we originally just output a single tensor?
        originally_single_tensor = False

        # Slight fixup for single tensor outputs.
        if not isinstance(cpu_outputs, (list, tuple)):
            originally_single_tensor = True
            cpu_outputs = [cpu_outputs]

        # Record metadata for our inputs & outputs, to later allocate in
        # permanent buffers.
        self.in_shapes = [i.shape for i in input]
        self.out_shapes = [o.shape for o in cpu_outputs]

        if _impl.isRunningOnIpu():
            outputs = [
                torch.zeros_like(o,
                                 device="ipu",
                                 requires_grad=o.requires_grad)
                for o in cpu_outputs
            ]
        else:
            outputs = cpu_outputs

        # End CPU host execution and show the JIT what the output looks like.
        outputs = torch.ops.poptorch.end_cpu_op(outputs)

        # Register this callback with poptorch so it knows what to call.
        poptorch_core.registerCPUCallBack(self, self._ID)

        # Just return one tensor if it was supposed to be just one.
        if originally_single_tensor:
            return outputs[0]

        return outputs


def identity_loss(x: "torch.Tensor", reduction: "str") -> "torch.Tensor":
    """Marks a tensor as being part of the loss calculation and, as such,
    will back-propagate through it in the PopTorch autograd.

    This function should be called on the (final) loss of a model so that
    it is used as the start of backpropagation. This is equivalent to calling
    ``x.backward()`` on a tensor ``x`` when running on the CPU.

    This function is necessary to combine multiple losses into a custom loss.
    It ensures that the tensor is part of the loss calculation and, as such,
    should be part of the backpropagation in PopTorch autograd.

    Multiple calls to ``identity_loss`` can be made inside the same model
    provided they are all dependant: all marked losses must be traceable
    into a single final tensor itself marked by a call to ``identity_loss``
    otherwise an error is raised.

    :param x: The calculated loss.
    :param reduction: Reduce the loss output as per PyTorch loss
        semantics. Supported values are:

        * ``"sum"``: Sum the losses.
        * ``"mean"``: Take the mean of the losses.
        * ``"none"``: Don't reduce the losses.

    :returns: The loss tensor with the specified reduction applied.
    """
    if reduction == "sum":
        return torch.ops.poptorch.identity_loss(x, 0)

    if reduction == "mean":
        return torch.ops.poptorch.identity_loss(x, 1)

    assert reduction == "none", "Unsupported reduction type!"
    return torch.ops.poptorch.identity_loss(x, 2)


def fps(src: "torch.Tensor",
        ptr: List[int],
        ratio: float = 0.5,
        random_start: bool = False) -> "torch.Tensor":
    """PopTorch implementation of the `torch_cluster` `fps` operator.

    This op is a sampling algorithm from the `"PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper, and iteratively samples the
    most distant point with regard to the rest points.

    :param src: Point feature matrix.
    :param ptr: Pointer vector which defines ranges of nodes assigned to a
        specific sample.
    :param ratio: The sampling ratio.
    :param random_start: If set to `False`, use the first node in `src` as
        the starting node.
    :returns: A tensor of `src` point indexes.
    """
    if not isinstance(src, torch.Tensor):
        raise _impl.createPoptorchError(
            f"`fps` must take a torch.tensor input. {type(src)} is "
            "not supported.")
    if not isinstance(ptr, list):
        raise _impl.createPoptorchError("`ptr` must be a list of integers.")
    if not len(ptr) >= 2:
        raise _impl.createPoptorchError(
            "`ptr` must containt at least 2 elements.")
    if not isinstance(ratio, float):
        raise _impl.createPoptorchError(
            f"`ratio` must be of float type. {type(ratio)} is not supported.")
    if not isinstance(random_start, bool):
        raise _impl.createPoptorchError(
            f"`random_start` must be of bool type. {type(random_start)} is "
            "not supported.")
    return torch.ops.poptorch.fps(src, ptr, ratio, random_start)


def nearest(x: "torch.Tensor",
            y: "torch.Tensor",
            batch_x: Optional[Union[List[int], "torch.Tensor"]] = None,
            batch_y: Optional[Union[List[int], "torch.Tensor"]] = None):
    """PopTorch implementation of the `torch_cluster` `nearest` operator.

    This op clusters points in `x` together which are nearest to a given query
    point in `y`.

    :param x: Node feature matrix.
    :param y: Node feature matrix.
    :param batch_x: Batch vector, which assigns each node to a specific
        sample. `batch_x` needs to be sorted.
    :param batch_y: Batch vector, which assigns each node to a specific
        sample. `batch_y` needs to be sorted.
    """

    if not isinstance(x, torch.Tensor):
        raise _impl.createPoptorchError(
            f"`nearest` must take a torch.tensor `x` input. {type(x)} is "
            "not supported.")
    if not isinstance(y, torch.Tensor):
        raise _impl.createPoptorchError(
            f"`nearest` must take a torch.tensor `y` input. {type(y)} is "
            "not supported.")

    batch_x = list() if batch_x is None else batch_x
    batch_y = list() if batch_y is None else batch_y

    batch_x_is_list = isinstance(batch_x, list)
    batch_y_is_list = isinstance(batch_y, list)
    batch_x_is_tensor = isinstance(batch_x, torch.Tensor)
    batch_y_is_tensor = isinstance(batch_y, torch.Tensor)

    if batch_x_is_list and batch_y_is_list:
        return torch.ops.poptorch.nearest_batch_list(x, y, batch_x, batch_y)
    if batch_x_is_tensor and batch_y_is_tensor:
        pass
    elif batch_x_is_list and batch_y_is_tensor:
        batch_x = torch.tensor(batch_x, dtype=batch_y.dtype)
    elif batch_x_is_tensor and batch_y_is_list:
        batch_y = torch.tensor(batch_y, dtype=batch_x.dtype)
    else:
        raise _impl.createPoptorchError(
            f"`batch_x` and `batch_y` must be torch.Tensors or lists while "
            f"`batch_x` is of type {type(batch_x)} and `batch_y` is of type "
            f"{type(batch_y)}.")
    return torch.ops.poptorch.nearest(x, y, batch_x, batch_y)


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
        :returns: ``self``, to support method chaining.
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
        :returns: ``self``, to support method chaining.
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
        :returns: ``self``, to support method chaining.
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

        :param value: An instance of :py:class:`~poptorch.MultiConvPlanType`.

        :returns: ``self``, to support method chaining.
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

        :param value: Number of tiles.
        :returns: ``self``, to support method chaining.
        """
        assert isinstance(value, int)
        self._per_conv_reserved_tiles = value
        return self

    def cycleBackOff(self, value: float) -> "poptorch.MultiConv":
        """Cycle back off proportion.

        :param value: Number between 0 and 1.
        :returns: ``self``, to support method chaining.
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
