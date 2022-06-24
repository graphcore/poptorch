=========
Changelog
=========

v2.6 (Poplar SDK 2.6)
=====================

New features
------------

- Improved performance of ``torch.gather`` in some cases where the index tensor has come from an ``expand`` or ``expand_as``.
- Improved error message when trying to apply bitwise ops to unsupported scalar types.
- Support for upsample bicubic mode.
- Support for ``zero_infinity`` in ``torch.nn.CTCLoss``.
- Experimental support for Torch's dispatcher as an alternative to ``torch.jit.trace()`` (see ref:`dispatcher-support`).
- Improved performance by compiling built-in custom ops at install time.

API changes
-----------

- Data loader: default process creation strategy for worker process changed from "spawn" to "forkserver".

Bug Fixes
---------

- Fixed remaining in-place operations on slices
- Fixed einsum transpose error
- Fixed floating point exception in ``torch.Tensor.exponential_`` and ``torch.distributions.Exponential``
- Improved support for ``torch.int16`` tensors.

v2.5 (Poplar SDK 2.5)
=====================

New features
------------

- Ignore missing values when reloading an Optimizer state.
- Support saving Optimizer states when compiling offline.
- Support for the following functions:

  * torch.var
  * torch.std
  * torch.var_mean
  * torch.std_mean

- Also save the random number generator's state and the seed when saving a model.

- Support for ``col2im`` (used by torch.nn.Fold).
- Improve error message of ``aten::index``, ``aten::index_put_`` when indexing with boolean tensor masks.
- Support for ``torch.argsort``.
- Support for ``torch.nn.RNN``.
- Add support for ``__repr__`` in PoplarExecutor.
- For models annotated with ``BeginBlock``, show the IPU blocks in ``repr(model)``.
- Improve implementation of torch.scatter_add
- Support for ``torch.nn.utils.weight_norm``
- Support for ``torch.randperm``
- Support for ``torch.nn.functional.cosine_similarity`` and ``torch.nn.CosineSimilarity``
- Support for ``torch.all``, ``torch.any``, ``torch.Tensor.all`` and ``torch.Tensor.any``
- Support for ``torch.Tensor.exponential_`` and ``torch.distributions.Exponential``

API changes
-----------

- Removed ``poptorch.AnchorMode``, ``poptorch.Options.anchorMode`` which were deprecated in favour of ``poptorch.OutputMode`` and ``poptorch.Options.outputMode`` respectively.

Bug Fixes
---------

- Fix thread safety issue in LogContext.
- Fix ``torch.clamp`` with integer tensors
- Fix in-place modification of slices
- Fix ``torch.index_put_`` when operating on slices
- Fix ``torch.chunk`` when dim size is indivisible by the specified number of chunks
- Fix cases where ``tensor.half()`` was in-place
- Fix tracing with half buffers
- Fix for loops with in-place ops
- Fix ``torch.flip`` with negative indices
- Fix masked assign when using tensor indexing syntax
- Fix some cases where use of serializedMatMul was ignored or resulted in errors.

v2.4 (Poplar SDK 2.4)
=====================

New features
------------
- Support for deepcopy functionality in ``poptorch.Options`` class
- Added functionality to add a name scope for each operator present in the module. This function is enabled by default. It can be disabled using ``poptorch.Options.disableModuleNamescope``.
- Support for a greater number of convolution and transpose convolution
  parameters including those which result in input/kernel/output truncation,
  either for inference (transpose) or gradient calculation.
- Migrated to PyTorch version 1.10.0
- Support for gradient clipping by norm in ``poptorch.optim`` optimizers
- Support saving and restoring internal optimiser state with PopTorch optimisers via ``optimizer.state_dict()`` and ``optimizer.load_state_dict()``
- Add ``removeBlocks`` function to remove block annotations from a Model / Layer.
- Support for CPU ops using ``poptorch.CPU``.
- Support for ``im2col`` (used by torch.nn.Unfold).
- Make optimizers work with LR schedulers.
- Switched to gold linker by default.

API changes
-----------
- Deprecated ``poptorch.Options.anchorMode`` in favour of ``poptorch.Options.outputMode``
- Deprecated ``poptorch.Options.defaultAnchorMode`` in favour of ``poptorch.Options.defaultOutputMode``
- Deprecated ``poptorch.AnchorMode`` in favour of ``poptorch.OutputMode``

Bug Fixes
---------
- Fixed incorrect gradient when using ``torch.nn.Embedding`` with ``padding_idx``

v2.3 (Poplar SDK 2.3)
=====================

New features
------------
- Support for ``torch.bitwise_and``, ``torch.bitwise_or``, ``torch.bitwise_xor``
- Support for ``torch.logical_and``, ``torch.logical_or``,
- Support K-dimensional NLLLoss, K-dimensional CrossEntropyLoss
- Support for non-default affine parameter flags in normalisation ops
- Support for ``torch.Tensor.T``
- Support for ``torch.bool`` in ``torch.zeros``, ``torch.zeros_like``, ``torch.ones``, ``torch.ones_like``
- Support for ``torch.scatter`` and its in-place variant
- Support for in-place modification to buffers on IPU
- Support for taking slices of scalars
- Support version of bilinear upsampling specifying intended output size instead of scale factors
- Add support for overlapping host IO on inputs via :py:func:`poptorch.set_overlap_for_input`.
- Add option for setting number of IO tiles via ``numIOTiles`` in ``poptorch.Options`` (required for :py:meth:`~poptorch.TensorLocationSettings.useIOTilesToLoad` and :py:func:`poptorch.set_overlap_for_input`.)
- Add method, :py:meth:`~poptorch.PoplarExecutor.cycleCount`, to determine the cycle count of the last model run.
- Improve PopTorch's parity with PyTorch's Softplus
- Improve implementation of torch.SiLU by using Poplar's Swish operator
- Additional support for operation overloads
- Add documentation on available memory proportion to incorporate embeddings and indexing operations
- Add documentation on how users can generate debug information
- Support replicated tensor sharding when running on multiple processes
- Allow selection for a non-constant x input.
- Support for ``enableConvDithering`` convolution option

Bug Fixes
---------
- Fix issue where PopTorch recalculated upsampling scales in fp16
- Fix issue where the last use of ``poptorch.set_available_memory`` would be pruned

API changes
-----------

- Default mean reduction strategies have changed from the deprecated PostAndLoss strategy to Post or Running
  based on optimiser accumulation type
- Mean reduction strategy can now be set via ``poptorch.Options.Training.setMeanAccumulationAndReplicationReductionStrategy``.
- Add warning that IPU-specific optimiser states cannot be read from the host, when calling ``get_state()`` on poptorch.optim optimisers

v2.2 (Poplar SDK 2.2)
=====================

New features
------------

- Migrated to PyTorch version 1.9.0
- Support for ``torch.roll``
- Support for ``torch.clone``
- Add modelName session option that can be passed to PopART
- Support List inputs to a model
- Tuples/Lists of constants can now be returned by a model
- Add ``enableProfiling`` convenience method in ``poptorch.Options`` to enable profile report generation
- Fix bug with ``torch.Tensor.repeat`` when applied to an input during training
- Fix bug with ``aten::to`` when applied to a constant used as an input to another node
- Improved error message when encountering untraceable types during compilation
- Support for ``torch.gather``. Please note: this operator is known to cause
  long compilation times. Consider using a onehot-based solution instead or
  `torch.index_select` if appropriate.
- Using a convolution layer op with the value of ``padding`` greater than or
  equal to `kernel_size`` is now supported.
- Support for Poplar recoverable and unrecoverable errors.
- Support for ``torch.flip``.
- Support for ``torch.Tensor.new_ones`` and ``torch.Tensor.new_zeros``

API changes
-----------

- Removed ``accumulationReductionType`` which was deprecated in 2.1 in favour of
  ``accumulationAndReplicationReductionType`` in ``poptorch.Options.Training``
- Removed ``runningVarianceAlwaysFloat`` which was deprecated in 2.1 and replaced by
  ``runningStatisticsAlwaysFloat`` in ``poptorch.Options.Precision``,

v2.1 (Poplar SDK 2.1)
=====================

New features
------------

- Support for ``torch.unbind``
- Add option to set `poptorch.Options` using options specified in a config file.
- Add ``mode=poptorch.DataLoaderMode.AsyncRebatched``
- Support for PopART name scopes via ``poptorch.NameScope``
- Add mixed precision automatic casting
- Support for ``torch.cross``
- Support for ``torch.functional.one_hot``
- Support for ``torch.int8`` data types
- Support for ``torch.median``
- Support for ``torch.index_select``
- Support for ``torch.scatter_add``
- Add ``poptorch.Options.Precision.enableFloatingPointExceptions`` to control floating point exception behavior
- Support for inplace changes to inputs.
- Add option to log the number of IPU cycles used in executing the main graph
- Support for ``torch.nn.GRU``
- Add automatic loss scaling option which can be enabled via ``poptorch.Options.Training.setAutomaticLossScaling``.
- Add ``poptorch.BlockFunction`` decorating for assigning an existing function
  to a block.
- Add mechanism for inspecting arbitrary tensors
- Add custom operator for CTC beam search decoding: ``poptorch.ctc_beam_search_decoder``
- Add a separate tensor variant (now default) to the SGD optimiser.
- Add a TensorFlow variant to the RMSProp optimiser.

API changes
-----------

- Removed ``Options.Popart`` which was deprecated in v2.0 and replaced with ``Options._Popart``
- Removed ``MultiConvPartialsType`` which was deprecated in v2.0
- Deprecated ``poptorch.Options.Training.accumulationReductionType`` in favour of ``poptorch.Options.Training.accumulationAndReplicationReductionType``
- Deprecated ``runningVarianceAlwaysFloat`` in favour of ``runningStatisticsAlwaysFloat`` in ``poptorch.Options.Precision``,
  as this new option computes both the running mean and variance in FP32 when this option is set to `True`.
- Use of SGD via PyTorch's or PopTorch's API now results in use of the new
  separate tensor variant by default. To revert to the previous default variant,
  use ``poptorch.optim.SGD`` with ``use_combined_accum=True``.

Known issues
------------

- Using a convolution layer op with the value of ``padding`` greater than or
  equal to `kernel_size`` results in an error when training. Use a constant pad
  layer instead of the excess padding prior to the convolution.

v2.0 (Poplar SDK 2.0)
=====================

New features
------------

- Support for the following activation functions:

  * torch.nn.acosh
  * torch.nn.asinh
  * torch.nn.atanh
  * torch.nn.Hardshrink
  * torch.nn.SiLU
  * torch.nn.Softplus
  * torch.nn.Softshrink
  * torch.nn.Threshold

- Support for the following random sampling operations:

  * torch.bernoulli
  * torch.distributions.Bernoulli

- Experimental support for torch.nn.CTCLoss
- Add Adam optimizer
- Support for ``torch.nn.AdaptiveAvgPool1d``, ``torch.nn.AdaptiveAvgPool3d``
- Migrated to PyTorch version 1.7.1
- Support for ``aten::index``, ``aten::index_put_``
- Support for ``torch.zeros_like``, ``torch.ones_like``
- Allow the user to specify which Optimizer attributes are constant or not.
- Allow the user to specify ``mode=poptorch.DataLoaderMode.Async`` in ``poptorch.DataLoader``
  constructor instead of explicitly creating an AsynchronousDataAccessor
- Support for ``torch.nn.EmbeddingBag``
- Support for ``torch.clamp_max`` and ``torch.clamp_min``
- Support for ``torch.min(tensor, dim=.*, keepdim=.*)`` and ``torch.max(tensor, dim=.*, keepdim=.*)`` overloads.
- Support for ``poptorch.isRunningOnIpu``. This function returns `True` when executing on IPU and `False` when executing
  the model outside IPU scope.
- Support for ``torch.amax`` and ``torch.amin``
- Support for attributes in custom ops.
- Support for precompilation and reloading exported executables (``poptorch.PoplarExecutor.compileAndExport`` and ``poptorch.load``)
- Support for slices with variable start index (slice size must be constant).
- Add ``ipuHardwareVersion`` function to read the version of the IPU hardware present on the system.
- Changed default targetd Ipu version for the model and offline compilation to `2`.
- Changed ``accumulationReductionType(reduction)`` option to now apply to replication reduction as well
- Add environment variable ``POPTORCH_CACHE_DIR``
- Support for ``torch.fmod``, and ``torch.remainder``
- Support for ``torch.addcdiv``
- Support for ``torch.bitwise_not``

API changes
-----------

- Deprecated ``Options.Popart``, ``Options._Popart`` may be used experimentally.

v1.0 (Poplar SDK 1.4)
=====================

New features
------------

- Support for torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d and torch.nn.InstanceNorm3d
- Fixed issue with torch.nn.GroupNorm where only 4-dimensional inputs could be used
- Replaced Adam with AdamW optimizer.
- Support for the following loss functions:

  * torch.nn.KLDivLoss
  * torch.nn.PoissonNLLLoss
  * torch.nn.HingeEmbeddingLoss
  * torch.nn.BCEWithLogitsLoss
  * torch.nn.SmoothL1Loss
  * torch.nn.SoftMarginLoss
  * torch.nn.CosineEmbeddingLoss
  * torch.nn.MarginRankingLoss
  * torch.nn.TripletMarginLoss
  * torch.nn.NLLLoss for aten::nll_loss2d

- Support for torch.optim.RMSprop optimizer
- Support for bool inputs to models
- Improved support for half type models and inputs.

  * Using a mix of float 16 and float 32 inputs is now supported. Please see
    the documentation for cases in which a model might use different types
    compared to when run natively with PyTorch.

- Support for serialized matrix multiplications
  (poptorch.serializedMatMul)
- Support for ``POPTORCH_IPU_MODEL_VERSION`` environment variable.
- Support for torch.cumsum
- Support for pipelined / phased / sharded execution.
- Add PoplarExecutor.compile() to compile the model without executing it.
- Use sphinx-build to generate the documentation.
- Use Miniconda as build environment.
- Support for torch.meshgrid
- Support for torch.cartesian_prod
- Optimized torch.matmul implementation with limitations

  * Fused its input 0's batch dimensions with the row dimension
    to avoid ReduceSum in its backward pass, for performance purpose

- Partial support for torch.einsum

  * Diagonals and ellipsis notation is unsupported

- Support for executable caching: poptorch.Options.enableExecutableCaching()
- Add optional title argument to poptorch.ipu_print_tensor
- Add len() method to poptorch.AsynchronousDataLoader
- Support for LAMB optimizer
- Support for recomputationCheckpoint()
- Support for torch.tensordot
- Support for rounding up the number of IPU used to allow models which
  specify of number of IPUs which is not a power of 2:
  poptorch.Options.autoRoundNumIPUs(True) NB, this will reserve but not use IPUs
  and so it is preferable to specify the model to use a number of IPUs which is
  a power of two
- Optimized torch.matmul implementation with limitations

  * Fused its input 0's batch dimensions with the row dimension
    to avoid ReduceSum in its backward pass, for performance purpose

- Support for multi-convolutions with poptorch.MultiConv
- Support for PopART batch serialization settings

  * These can be set via poptorch.Options().Popart.set()

- Support for PopVision System Analyser added: tracing can be enabled by setting ``PVTI_OPTIONS='{"enable":"true"}'``

Known issues
------------

- Race condition in ``poptorch.DataLoader`` when using several workers resulting in the iteration sometimes finishing one element early.

  * Workaround: set ``num_workers`` to 0 or 1.

- ``poptorch.custom_op()`` doesn't allow the user to set attributes.

  * Workaround: hardcode the attributes in the custom operation or pass them as regular inputs.

- Graphs containing block annotations (``poptorch.Block`` or ``poptorch.BeginBlock``) cannot be exported using ``torch.save()``

  * Workaround: Make a soft copy of the model that doesn't contain Blocks and use it to save /load the weights. (The weights should be shared between the two models).

- Lists of tensors are not supported as inputs.

  * Workaround: Use tuples instead.

    .. code-block:: python

      # Use a tuple
      assert inference_model((t1, t2)) # instead of [t1, t2]

v0.1 (Poplar SDK 1.3)
=====================

New features
------------

- PopTorch now exposes PopART anchor options to choose how much data to return from a model. These
  are passed into the model wrapper via anchor_mode. options are Sum, All, Final and EveryN.
- Support for batched LSTM and batch first
- An Options object can now be passed to poptorch.trainingModel / poptorch.inferenceModel to configure the session and select IPUs
- The 'profile' option has been removed, instead profiling can be enabled by
  setting the environment variable ``POPLAR_ENGINE_OPTIONS='{autoReport.all:true, autoReport.directory:.}'``
- Support for ``POPTORCH_IPU_MODEL`` and ``POPTORCH_WAIT_FOR_IPU`` environment variables.
- Support for the torch comparisons operations:

  * torch.eq
  * torch.ge
  * torch.gt
  * torch.le
  * torch.lt
  * torch.max
  * torch.min
  * torch.ne
  * torch.isnan
  * torch.topk
  * torch.min and torch.max only support (tensor, tensor) and (tensor) overloads.
    They do not support the (tensor, dim=, keepdim=) overload.
  * torch.topk only supports sorted=False and Largest=True

- Automatically synchronise the weights back to the Host after using the IPU for training. (i.e no need to explicitly call copyWeightsToHost() anymore)
- Support for non-linear activations torch.nn.PReLU and torch.nn.Hardtanh
- Support for Adam optimizer.
- Support for half type models and inputs.

  * Models that require operations on input tensors of mixed precision are not currently supported.
    For example:

    .. code-block:: python

        def forward(self, x, y):
          x # Half
          y # Float32
          return x + y # Not supported.

- Support for ``tensor.fill_``, ``torch.full``, ``torch.full_like``

- Support for user provided custom operations. See PopART documentation for information on
  how to write them. They are exposed by `poptorch.custom_op` this takes in a list of
  input tensors, strings for the PopART op name and domain, the domain version, and
  a list of tensors the same shape and size as the expected output tensors. This is to
  ensure the pytorch trace remains valid as it traces on CPU so won't actually execute
  the operation when building the graph.

- Support for torch.nn.Conv1D / torch.nn.Conv2D / torch.nn.Conv3D

- Support for torch.nn.Upsample ('nearest' mode only)

- Support for tensor.size

- Support for the following random sampling operations.

  * ``torch.rand``
  * ``torch.uniform_``
  * ``torch.distributions.Uniform``
  * ``torch.randn``
  * ``torch.normal``
  * ``torch.normal_``

  For repeatable random number generation use the `randomSeed` method of `poptorch.Options`

- Support for torch.clamp

- Adds poptorch.DataLoader

- Adds optimized poptorch.AsynchronousDataAccessor which allows for a dataloader to be offloaded to a background thread asynchronously.

- Support for torch.norm

- Upgraded from torch 1.5.0 to torch 1.6.0

- Experimental support for single host distributed execution

- Add torch.where and tensor.masked_fill
