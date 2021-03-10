=========
Changelog
=========

v2.1 (Poplar SDK 2.1)
=====================

New features
------------


v2.0 (Poplar SDK 2.0)
=====================

New features
------------

- Added support for the following activation functions:

  * torch.nn.acosh
  * torch.nn.asinh
  * torch.nn.atanh
  * torch.nn.Hardshrink
  * torch.nn.SiLU
  * torch.nn.Softplus
  * torch.nn.Softshrink
  * torch.nn.Threshold

- Add support for the following random sampling operations:

  * torch.bernoulli
  * torch.distributions.Bernoulli

- Add experimental support for torch.nn.CTCLoss
- Added Adam optimizer
- Added support for ``torch.nn.AdaptiveAvgPool1d``, ``torch.nn.AdaptiveAvgPool3d``
- Migrated to PyTorch version 1.7.1
- Add support for ``aten::index``, ``aten::index_put_``
- Add support for ``torch.zeros_like``, ``torch.ones_like``
- Allow the user to specify which Optimizer attributes are constant or not.
- Allow the user to specify ``mode=poptorch.DataLoaderMode.Async`` in ``poptorch.DataLoader``
  constructor instead of explicitly creating an AsynchronousDataAccessor
- Add support for ``torch.nn.EmbeddingBag``
- Added support for ``torch.clamp_max`` and ``torch.clamp_min``
- Add support for ``torch.min(tensor, dim=.*, keepdim=.*)`` and ``torch.max(tensor, dim=.*, keepdim=.*)`` overloads.
- Add support for ``poptorch.isRunningOnIpu``. This function returns `True` when executing on IPU and `False` when executing
  the model outside IPU scope.
- Add support for ``torch.amax`` and ``torch.amin``
- Add support for attributes in custom ops.
- Add support for precompilation and reloading exported executables (``poptorch.PoplarExecutor.compileAndExport`` and ``poptorch.load``)
- Add support for slices with variable start index (slice size must be constant).
- Add ``ipuHardwareVersion`` function to read the version of the IPU hardware present on the system.
- Changed default targetd Ipu version for the model and offline compilation to `2`.
- Changed ``accumulationReductionType(reduction)`` option to now apply to replication reduction as well
- Add environment variable ``POPTORCH_CACHE_DIR``
- Deprecated ``Options.Popart``, ``Options._Popart`` may be used experimentally.
- Add support for ``torch.fmod``, and ``torch.remainder``
- Add support for ``torch.addcdiv``

v1.0 (Poplar SDK 1.4)
=====================

New features
------------

- Added support for torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d and torch.nn.InstanceNorm3d
- Fixed issue with torch.nn.GroupNorm where only 4-dimensional inputs could be used
- Replaced Adam with AdamW optimizer.
- Added support for the following loss functions:

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

- Added support for torch.optim.RMSprop optimizer
- Added support for bool inputs to models
- Improved support for half type models and inputs.

  * Using a mix of float 16 and float 32 inputs is now supported. Please see
    the documentation for cases in which a model might use different types
    compared to when run natively with PyTorch.

- Added support for serialized matrix multiplications
  (poptorch.serializedMatMul)
- Add support for ``POPTORCH_IPU_MODEL_VERSION`` environment variable.
- Added support for torch.cumsum
- Add support for pipelined / phased / sharded execution.
- Add PoplarExecutor.compile() to compile the model without executing it.
- Use sphinx-build to generate the documentation.
- Use Miniconda as build environment.
- Added support for torch.meshgrid
- Added support for torch.cartesian_prod
- Optimized torch.matmul implementation with limitations

  * Fused its input 0's batch dimensions with the row dimension
    to avoid ReduceSum in its backward pass, for performance purpose

- Added partial support for torch.einsum

  * Diagonals and ellipsis notation is unsupported

- Add support for executable caching: poptorch.Options.enableExecutableCaching()
- Add optional title argument to poptorch.ipu_print_tensor
- Add len() method to poptorch.AsynchronousDataLoader
- Added support for LAMB optimizer
- Add support for recomputationCheckpoint()
- Added support for torch.tensordot
- Add support for rounding up the number of IPU used to allow models which
  specify of number of IPUs which is not a power of 2:
  poptorch.Options.autoRoundNumIPUs(True) NB, this will reserve but not use IPUs
  and so it is preferable to specify the model to use a number of IPUs which is
  a power of two
- Optimized torch.matmul implementation with limitations

  * Fused its input 0's batch dimensions with the row dimension
    to avoid ReduceSum in its backward pass, for performance purpose

- Added support for multi-convolutions with poptorch.MultiConv
- Added support for PopART batch serialization settings

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
- Add support for batched LSTM and batch first
- An Options object can now be passed to poptorch.trainingModel / poptorch.inferenceModel to configure the session and select IPUs
- The 'profile' option has been removed, instead profiling can be enabled by
  setting the environment variable ``POPLAR_ENGINE_OPTIONS='{autoReport.all:true, autoReport.directory:.}'``
- Add support for ``POPTORCH_IPU_MODEL`` and ``POPTORCH_WAIT_FOR_IPU`` environment variables.
- Adds support for the torch comparisons operations:

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
- Adds support for non-linear activations torch.nn.PReLU and torch.nn.Hardtanh
- Adds support for Adam optimizer.
- Adds support for half type models and inputs.

  * Models that require operations on input tensors of mixed precision are not currently supported.
    For example:

    .. code-block:: python

        def forward(self, x, y):
          x // Half
          y // Float32
          return x + y // Not supported.

- Support for ``tensor.fill_``, ``torch.full``, ``torch.full_like``

- Adds support for user provided custom operations. See PopART documentation for information on
  how to write them. They are exposed by `poptorch.custom_op` this takes in a list of
  input tensors, strings for the PopART op name and domain, the domain version, and
  a list of tensors the same shape and size as the expected output tensors. This is to
  ensure the pytorch trace remains valid as it traces on CPU so won't actually execute
  the operation when building the graph.

- Adds support for torch.nn.Conv1D / torch.nn.Conv2D / torch.nn.Conv3D

- Adds support for torch.nn.Upsample ('nearest' mode only)

- Adds support for tensor.size

- Adds support for the following random sampling operations.

  * ``torch.rand``
  * ``torch.uniform_``
  * ``torch.distributions.Uniform``
  * ``torch.randn``
  * ``torch.normal``
  * ``torch.normal_``

  For repeatable random number generation use the `randomSeed` method of `poptorch.Options`

- Adds support for torch.clamp

- Adds poptorch.DataLoader

- Adds optimized poptorch.AsynchronousDataAccessor which allows for a dataloader to be offloaded to a background thread asynchronously.

- Adds support for torch.norm

- Upgraded from torch 1.5.0 to torch 1.6.0

- Added experimental support for single host distributed execution

- Added torch.where and tensor.masked_fill
