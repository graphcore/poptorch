# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import enum
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from poptorch.poptorch_core import *
import poptorch.poptorch_core as poptorch_core

begin_ipu_block = torch.ops.poptorch.begin_ipu_block
end_ipu_block = torch.ops.poptorch.end_ipu_block
ipu_print_tensor = torch.ops.poptorch.ipu_print_tensor

# Create a poptorch logger which outputs to the console INFO messages and above
logger = logging.getLogger("poptorch")
if os.environ.get("POPTORCH_LOG_LEVEL") in ["DEBUG", "TRACE", "TRACE_ALL"]:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
console = logging.StreamHandler()
formatter = logging.Formatter('%(name)s:%(levelname)s: %(message)s')
console.setFormatter(formatter)
console.setLevel(logging.DEBUG)
logger.addHandler(console)


def identity_loss(x, reduction="none"):
    if (reduction == "sum"):
        return torch.ops.poptorch.identity_loss(x, 0)

    if (reduction == "mean"):
        return torch.ops.poptorch.identity_loss(x, 1)

    assert reduction == "none", "Unsupported reduction type!"
    return torch.ops.poptorch.identity_loss(x, 2)


def _convertOptimizerToDict(optimizer):
    if len(optimizer.param_groups) != 1:
        print(
            "Error: Poptorch currently only supports one parameter group! (all parameters)"
        )
        exit()

    return {
        "lr": (optimizer.param_groups[0]["lr"], False),
        "momentum": (optimizer.param_groups[0]["momentum"], False),
        "weight_decay": (optimizer.param_groups[0]["weight_decay"], False),
        "dampening": (optimizer.param_groups[0]["dampening"], False)
    }


class _OptionsDict:
    """Safe dictionary to store options: only keys which have been passed to the constructor can later be updated.
    """

    def __init__(self, **default_values):
        self._values = default_values

    def set(self, **kwargs):
        for option, value in kwargs.items():
            assert option in self._values, "Invalid option %s, valid options are %s" % (
                option, self._values.keys())
            assert type(value) == type(
                self._values[option]
            ), "Unexpected type %s for option %s. Expected %s" % (
                type(value), option, type(self._values[option]))
            self._values[option] = value

    def createOrSet(self, **kwargs):
        for option, value in kwargs.items():
            if option in self._values:
                self.set(option=value)
            else:
                self._values[option] = value

    def __getattr__(self, option):
        assert option in self._values, "Invalid option %s, valid options are %s" % (
            option, self._values.keys())
        return self._values[option]

    def update(self, other):
        assert not set(self._values.keys()).intersection(
            other), "Can't merge dictionaries, they have some keys in common"
        other.update(self._values)
        return other

    def __call__(self, option):
        assert option in self._values, "Invalid option %s, valid options are %s" % (
            option, self._values.keys())
        return self._values[option]


class _JitOptions(_OptionsDict):
    """Options related to Pytorch's JIT
    """

    def __init__(self):
        super().__init__(trace_model=True)

    def traceModel(self, trace_model):
        """
        If True: use torch.jit.trace
        If False: use torch.jit.script

        Trace model is enabled by default.
        """
        self.set(trace_model=trace_model)
        return self


class _TrainingOptions(_OptionsDict):
    """Options specific to model training.
    """

    def __init__(self):
        super().__init__(gradient_accumulation=1)

    def gradientAccumulation(self, gradient_accumulation):
        self.set(gradient_accumulation=gradient_accumulation)
        return self


class _PopartOptions:
    """Options specific to the Popart backend.
    Only for advanced users.
    """

    def __init__(self):
        self.options = {}

    def set(self, key, value):
        self.options[key] = value
        return self


class AnchorMode(enum.IntEnum):
    """
    All: Return a result for each batch.
    Sum: Return the sum of all the batches
    Final: Return the last batch.
    EveryN: Return every N batches. N is passed in as |anchor_return_period|
    Default: "All" for inference, "Final" for training.
    """
    Final = 0
    EveryN = 1
    All = 2
    Sum = 3
    Default = 4


class ConnectionType(enum.IntEnum):
    """
    - Always: Attach to the IPU from the start (Default).
    - OnDemand: Wait until the compilation is complete and the executable is ready to be run to attach to the IPU.
    - Never: Never try to attach to an IPU. (Useful for offline compilation, but trying to run an executable will raise an exception).
    """
    Always = 0
    OnDemand = 1
    Never = 2


class SyncPattern(enum.IntEnum):
    Full = 0
    SinglePipeline = 1
    PingPong = 2


class Options(_OptionsDict):
    def __init__(self):
        self._jit = _JitOptions()
        self._training = _TrainingOptions()
        self._popart = _PopartOptions()

        super().__init__(
            replication_factor=1,
            device_iterations=1,
            log_dir=".",
            profile=False,
            anchor_mode=AnchorMode.Default.value,
            anchor_return_period=1,
            use_model=False,
            connection_type=ConnectionType.Always.value,
            sync_pattern=SyncPattern.Full.value,
        )

    @property
    def Jit(self):
        """Options specific to PyTorch's JIT."""
        return self._jit

    @property
    def Training(self):
        """Options specific to training."""
        return self._training

    @property
    def Popart(self):
        """Options specific to the Popart backend.
        (Advanced users only).
        """
        return self._popart

    def deviceIterations(self, device_iterations):
        """Number of iterations run on the device per execution (Default: 1)"""
        self.set(device_iterations=device_iterations)
        return self

    def enablePipelining(self, enable_pipelining):
        """Enable pipelining of virtual graphs (Default: False if 1 IPU used, True otherwise)"""
        self.createOrSet(enable_pipelining=enable_pipelining)
        return self

    def replicationFactor(self, replication_factor):
        """Number of model replications (Default: 1).

        E.g. if your model uses 1 IPU, a
        replication factor of 2 will use 2 IPUs. If your model is
        pipelined across 4 IPUs, a replication factor of 4 will use 16 IPUs
        total.
        """
        self.set(replication_factor=replication_factor)
        return self

    def logDir(self, log_dir):
        """Where to save log files (Default: Current directory)"""
        self.set(log_dir=log_dir)
        return self

    def profile(self, profile):
        """Enable profiling (Default: False)"""
        self.set(profile=profile)
        return self

    def useIpuModel(self, use_model):
        """Use the IPU model or physical hardware.

        Default: False (Real Hardware)
        This setting takes precedence over the POPTORCH_IPU_MODEL environment variable.
        """
        self.set(use_model=use_model)
        return self

    def connectionType(self, connection_type):
        """set the IPU connection type to one of:
        - Always: Attach to the IPU from the start (Default).
        - OnDemand: Wait until the compilation is complete and the executable is ready to be run to attach to the IPU.
        - Never: Never try to attach to an IPU. (Useful for offline compilation, but trying to run an executable will raise an exception).
        """
        assert isinstance(connection_type, ConnectionType)
        self.set(connection_type=connection_type.value)
        return self

    def syncPattern(self, sync_pattern):
        """set the IPU SyncPatter to one of:
        - Full
        - SinglePipeline
        - PingPong
        """
        assert isinstance(sync_patter, SyncPattern)
        self.set(sync_pattern=sync_pattern.value)
        return self

    def useIpuId(self, ipu_id):
        """ Use the specified IPU id as provided by gc-info.

        The number of IPUs associated with the id must be equal to the number of IPUs used by your grpah multiplied by the replication factor.

        E.g. if your model uses 1 IPU and the replication factor is 2 you will need to provide an id with 2 IPUs.
        If your model is pipelined across 4 IPUs, the replication factor is 4, you will need to provide an id containing 16 IPUs.
        total.
        """
        assert isinstance(ipu_id, int)
        self.createOrSet(ipu_id=ipu_id)
        return self

    def useOfflineIpuTarget(self, ipu_version=1):
        """Create an offline IPU target that can only be used for offline compilation.

        Note: the offline IPU target cannot be used if the IPU model is enabled.
        """
        self.connectionType(ConnectionType.Never)
        self.createOrSet(ipu_version=ipu_version)
        return self

    def anchorMode(self, anchor_mode, anchor_return_period=None):
        """ How much data to return from a model

        Args:
            anchor_mode:
                All: Return a result for each batch.
                Sum: Return the sum of all the batches
                Final: Return the last batch.
                EveryN: Return every N batches. N is passed in as |anchor_return_period|
                Default: "All" for inference, "Final" for training.
        """
        assert isinstance(anchor_mode, AnchorMode)

        # Check the anchor return period makes sense.
        if anchor_mode == AnchorMode.EveryN:
            assert anchor_return_period and anchor_return_period > 0, "EveryN anchor must have anchor_return_period set to valid positive integer"
        elif anchor_return_period:
            logging.info(
                "Anchor return period argument ignored with anchor_mode set to %s"
                % anchor_mode)

        self.set(anchor_mode=anchor_mode.value,
                 anchor_return_period=anchor_return_period or 1)
        return self

    def defaultAnchorMode(self):
        """Return True if the anchor_mode is currently set to Default, False otherwise."""
        return self.anchor_mode == AnchorMode.Default

    def toDict(self):
        """ Merge all the options, except for the Jit ones, into a single dictionary to be serialised and passed to the cpp side."""
        assert not self.defaultAnchorMode(
        ), "An anchor mode must be picked before serialisation"
        out = {}
        out.update(self._popart.options)
        out = self.update(out)
        out = self._training.update(out)
        return out


class IPU(nn.Module):
    def __init__(self, ipu_id, layer_to_call=None):
        super().__init__()

        self.ipu_id = ipu_id
        self.layer_to_call = layer_to_call

    def __enter__(self):
        begin_ipu_block(self.ipu_id)

    def __exit__(self, type, value, traceback):
        end_ipu_block()

    def __call__(self, *input, **kwargs):
        begin_ipu_block(self.ipu_id)
        out = self.layer_to_call(*input, **kwargs)
        return out


class _Args:
    def __init__(self, model, args, kwargs, training):
        # Combine args and kwargs:
        self._args = []
        fn = model.__call__ if training else model.forward
        varnames = fn.__code__.co_varnames
        # Remove 'self'
        assert varnames[0] == 'self'
        argcount = fn.__code__.co_argcount
        varnames = varnames[1:argcount]
        argcount -= 1
        assert len(args) + len(
            kwargs
        ) <= argcount, "Too many arguments provided: expected %s (%d) but got %d" % (
            varnames, len(varnames), len(args) + len(kwargs))
        defaults = fn.__defaults__ or []
        first_optional = len(varnames) - len(defaults)
        none_passed = []
        for i, name in enumerate(varnames):
            if i < len(args):
                self._args.append(args[i])
                assert name not in kwargs, "Parameter %s was passed more than once" % name
            elif name in kwargs:
                assert not none_passed, "Torch doesn't support passing tensors after the following parameters have defaulted to None. %s" % ", ".join(
                    none_passed)
                self._args.append(kwargs[name])
            else:
                assert i >= first_optional, "Mandatory parameter %s missing" % name
                value = defaults[i - first_optional]
                if value == None:
                    none_passed.append("%s (%d)" % (name, i))
                if not none_passed:
                    self._args.append(value)

        self._varnames = varnames

    def _forEach(self, data, fn):
        if isinstance(data, (tuple, list)):
            return type(data)(self._forEach(d, fn) for d in data)
        elif isinstance(data, dict):
            return {
                key: self._forEach(value, fn)
                for key, value in data.items()
            }
        else:
            return fn(data)

    def _forEachMatched(self, data, condition, doOnTrue, conditionMatches):
        if isinstance(data, (tuple, list)):
            return type(data)(
                self._forEachMatched(d, condition, doOnTrue, conditionMatches)
                for d in data)
        elif isinstance(data, dict):
            return {
                key: self._forEachMatched(value, condition, doOnTrue,
                                          conditionMatches)
                for key, value in data.items()
            }
        else:
            if condition(data):
                conditionMatches.setTrue()
                return doOnTrue(data)
            else:
                return data

    def forEachMatchedAtLeastOnce(self, condition, doOnTrue=None):
        class ConditionMatches(object):
            def __init__(self):
                self._matches = False

            def __bool__(self):
                return self._matches

            def setTrue(self):
                self._matches = True

        matches = ConditionMatches()
        self._args = self._forEachMatched(self._args, condition, doOnTrue,
                                          matches)
        return bool(matches)

    def forEach(self, fn):
        self._args = self._forEach(self._args, fn)

    def asTuple(self):
        return tuple(self._args)


class _PoplarExecutor:
    def __init__(self, model, options, training, optimizer={}):
        self.executable = None
        self.options = options
        self.model = model
        self.training = training
        self.optimizer = optimizer
        self.new_optimizer = optimizer
        self.warned_not_contiguous_input = False
        self.dirty_host_weights = False
        if self.training:
            m = self.model.model

            class WrappedModel(type(m)):
                def copyWeightsToHostIfNeeded(self_model):
                    """ Return True if the weights on the host were dirty and have been updated.
                    Return False if the weights were already up to date.
                    """
                    if self.dirty_host_weights:
                        logger.debug("Implicit copyWeightsToHost()")
                        self.copyWeightsToHost()
                        self.dirty_host_weights = False
                        return True
                    return False

                def __call__(self_model, *args, **kwargs):
                    # If the model has been trained on the IPU: update the host side weights
                    self_model.copyWeightsToHostIfNeeded()
                    return self.model.real_model_call(*args, **kwargs)

                def named_parameters(self_model, *args, **kwargs):
                    self_model.copyWeightsToHostIfNeeded()
                    return super().named_parameters(*args, **kwargs)

            # __call__ is an attribute, not a method, unfortunately we cannot just
            # replace it in the model object: we have to create a wrapper class and change the object's class.
            m.__class__ = WrappedModel

    # Copy weights from the device into the memory of the model given on wrapper creation.
    def copyWeightsToHost(self):
        copyWeightsToHost_impl(self.executable)

    # Write from host memory to IPU memory. This is done automatically on compilation so should be rarely used.
    def copyWeightsToDevice(self):
        copyWeightsToDevice_impl(self.executable)

    def setOptimizer(self, optimizer):
        self.new_optimizer = optimizer

    def __call__(self, *args, **kwargs):
        # Convert single tensor to tuple.
        in_tensors = _Args(self.model, args, kwargs, self.training)

        if in_tensors.forEachMatchedAtLeastOnce(
                condition=lambda t: not t.is_contiguous(),
                doOnTrue=lambda t: t.contiguous()):
            if not self.warned_not_contiguous_input:
                logger.warning(
                    "At least one input tensor is not contiguous: " +
                    "non-contiguous tensors will be converted.")
                self.warned_not_contiguous_input = True

        if self.executable == None:
            logger.info(
                "First time call to model will invoke poplar compilation. " +
                str(self.options.device_iterations) + " " + str(self.training))

            # Input will be in form of [BatchSize* BatchPerStep, ...] so we should slice it up so we compile by the batch size alone.
            extra_poplar_batch_dims = self.options.device_iterations * \
                self.options.replication_factor * self.options.Training.gradient_accumulation

            # There are two concepts of batch size. First is the "model" batch size then there is the
            # concept of batching at the popart level. Here we divide by the popart batch size so the
            # trace "sees" the model batch size but when we call execute we pass the full batch and popart
            # will partition it up.
            in_tensors_trace_view = _Args(self.model, args, kwargs,
                                          self.training)
            in_tensors_trace_view.forEach(lambda t: t.narrow(
                0, 0,
                t.size()[0] // extra_poplar_batch_dims) if isinstance(
                    t, torch.Tensor) else t)

            # Compile the poplar executable based on the batchsize.
            if self.options.Jit.trace_model:
                logger.info('Compiling the model using tracing')
                n = torch.jit.trace(self.model,
                                    in_tensors_trace_view.asTuple())

                self.executable = compileWithTrace(
                    n._c, in_tensors_trace_view.asTuple(),
                    self.options.toDict(), self.training, self.optimizer)
            else:
                logger.info('Compiling the model using scripting')
                n = torch.jit.script(self.model)
                graphInputs = list(n.graph.inputs())
                for graphInput, argIn in zip(graphInputs[1:],
                                             in_tensors_trace_view.asTuple()):
                    if isinstance(argIn, torch.Tensor):
                        graphInput.inferTypeFrom(argIn)

                self.executable = compileWithScript(
                    n._c, n.graph, in_tensors_trace_view.asTuple(),
                    self.options.toDict(), self.training)

        if self.options.connectionType == ConnectionType.Never:
            logger.info(
                "Compilation complete and ConnectionType.Never selected: returning"
            )
            return

        # If this is an inference model: check if the same model is not being trained on a different IPU.
        # If it is: make sure the weights are updated.
        if not self.training:
            copyWeightsToHostIfNeeded = getattr(
                self.model, "copyWeightsToHostIfNeeded", None)
            if callable(copyWeightsToHostIfNeeded):
                if copyWeightsToHostIfNeeded():
                    # Weights have now been updated on the Host: copy them to the second IPU.
                    logger.debug("Implicit copyWeightsToDevice()")
                    self.copyWeightsToDevice()

        # Execute the poplar executable with the full size (batch * device interations)
        if self.new_optimizer and self.new_optimizer != self.optimizer:
            self.optimizer = self.new_optimizer
            output = execute(self.executable, in_tensors.asTuple(),
                             _convertOptimizerToDict(self.optimizer))
        else:
            output = execute(self.executable, in_tensors.asTuple(), {})

        if self.training:
            self.dirty_host_weights = True

        if len(output) > 1:
            return output
        else:
            return output[0]


def trainingModel(model, options=None, loss=None, optimizer=None):
    options = options or Options()
    if options.defaultAnchorMode():
        # In training it makes sense to see only the last result, by default.
        options.anchorMode(AnchorMode.Final)
    if not optimizer:
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    optimizer = _convertOptimizerToDict(optimizer)

    class ModelTrainingWrapper(nn.Module):
        def __init__(self, model, loss=None):
            super().__init__()
            self.model = model
            self.loss = loss
            # Store the real __call__ method before _PoplarExecutor wraps it
            self.real_model_call = model.__call__

        def __call__(self, args, loss_inputs):
            output = self.real_model_call(args)

            if self.loss:
                loss = self.loss(output, loss_inputs)
                return output, loss

            return output

    wrappedModel = ModelTrainingWrapper(model, loss)
    return _PoplarExecutor(model=wrappedModel,
                           options=options,
                           training=True,
                           optimizer=optimizer)


def inferenceModel(model, options=None):
    options = options or Options()
    if options.defaultAnchorMode():
        # In inference it makes sense to see all the results, by default.
        options.anchorMode(AnchorMode.All)
    return _PoplarExecutor(model=model, options=options, training=False)


def propagateInputShapes(graph, dummyInputs):
    for graphInput, dummyInput in zip(graph.inputs(), dummyInputs):
        graphInput.inferTypeFrom(dummyInput)
    poptorch_core.propagateInputShapes(graph)
