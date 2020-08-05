# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.optim as optim
import poptorch.poptorch_core as poptorch_core

from . import enums
from .logging import logger


def convertOptimizerToDict(optimizer):
    assert len(optimizer.param_groups) == 1, (
        "Poptorch currently only "
        "supports one parameter group! (all parameters)")

    learning_rate = optimizer.param_groups[0]["lr"]
    weight_decay = optimizer.param_groups[0]["weight_decay"]

    if isinstance(optimizer, optim.SGD):
        momentum = optimizer.param_groups[0]["momentum"]
        dampening = optimizer.param_groups[0]["dampening"]
        # We will default momentum, weight decay, and dampening, to be
        # constant if they are set to zero.
        return {
            "optimizerType": enums.OptimizerType.SGD,
            "lr": (learning_rate, False),
            "momentum": (momentum, momentum == 0.0),
            "weight_decay": (weight_decay, weight_decay == 0.0),
            "dampening": (dampening, dampening == 0.0)
        }
    if isinstance(optimizer, optim.Adam):
        beta1 = optimizer.param_groups[0]["betas"][0]
        beta2 = optimizer.param_groups[0]["betas"][1]
        eps = optimizer.param_groups[0]["eps"]

        assert not optimizer.param_groups[0]["amsgrad"], (
            "Only non-amsgrad "
            "Adam optimizers are supported.")
        return {
            "optimizerType": enums.OptimizerType.ADAM,
            "lr": (learning_rate, False),
            "beta1": (beta1, False),
            "beta2": (beta2, False),
            "weight_decay": (weight_decay, weight_decay == 0.0),
            "eps": (eps, eps == 1e-08)
        }

    assert False, "Unsupported optimizer type. Types supported %s" % str(
        list(enums.OptimizerType))
    return None


class OptionsDict:
    """Safe dictionary to store options: only keys which have been passed to
    the constructor can later be updated.
    """

    def __init__(self, **default_values):
        self._values = default_values

    def set(self, **kwargs):
        for option, value in kwargs.items():
            assert option in self._values, ("Invalid option %s, valid options"
                                            " are %s") % (option,
                                                          self._values.keys())
            assert isinstance(
                value, type(self._values[option])
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
        assert option in self._values, ("Invalid option %s, "
                                        "valid options are %s") % (
                                            option, self._values.keys())
        return self._values[option]

    def update(self, other):
        assert not set(self._values.keys()).intersection(
            other), "Can't merge dictionaries, they have some keys in common"
        other.update(self._values)
        return other

    def __call__(self, option):
        assert option in self._values, ("Invalid option %s, "
                                        "valid options are %s") % (
                                            option, self._values.keys())
        return self._values[option]


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
        assert len(args) + len(kwargs) <= argcount, (
            "Too many arguments provided: expected %s (%d) "
            "but got %d") % (varnames, len(varnames), len(args) + len(kwargs))
        defaults = fn.__defaults__ or []
        first_optional = len(varnames) - len(defaults)
        none_passed = []
        for i, name in enumerate(varnames):
            if i < len(args):
                self._args.append(args[i])
                assert name not in kwargs, ("Parameter %s was passed more "
                                            "than once") % name
            elif name in kwargs:
                assert not none_passed, (
                    "Torch doesn't support passing tensors"
                    " after the following parameters have defaulted to None."
                    " %s") % ", ".join(none_passed)
                self._args.append(kwargs[name])
            else:
                assert i >= first_optional, ("Mandatory parameter %s "
                                             "missing") % name
                value = defaults[i - first_optional]
                if value is None:
                    none_passed.append("%s (%d)" % (name, i))
                if not none_passed:
                    self._args.append(value)

        self._varnames = varnames

    def _forEach(self, data, fn):
        if isinstance(data, (tuple, list)):
            return type(data)(self._forEach(d, fn) for d in data)
        if isinstance(data, dict):
            return {
                key: self._forEach(value, fn)
                for key, value in data.items()
            }
        return fn(data)

    def _forEachMatched(self, data, condition, doOnTrue, conditionMatches):
        if isinstance(data, (tuple, list)):
            return type(data)(
                self._forEachMatched(d, condition, doOnTrue, conditionMatches)
                for d in data)
        if isinstance(data, dict):
            return {
                key: self._forEachMatched(value, condition, doOnTrue,
                                          conditionMatches)
                for key, value in data.items()
            }
        if condition(data):
            conditionMatches.setTrue()
            return doOnTrue(data)
        return data

    def forEachMatchedAtLeastOnce(self, condition, doOnTrue=None):
        class ConditionMatches:
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


class PoplarExecutor:
    def __init__(self, model, options, training, optimizer=None):
        self.executable = None
        self.options = options
        self.model = model
        self.training = training
        self.optimizer = optimizer or {}
        self.new_optimizer = optimizer or {}
        self.warned_not_contiguous_input = False
        self.dirty_host_weights = False
        if self.training:
            m = self.model.model
            parent = self

            class WrappedModel(type(m)):
                def copyWeightsToHostIfNeeded(self):
                    """ Return True if the weights on the host were dirty and
                    have been updated.
                    Return False if the weights were already up to date.
                    """
                    if parent.dirty_host_weights:
                        logger.debug("Implicit copyWeightsToHost()")
                        parent.copyWeightsToHost()
                        parent.dirty_host_weights = False
                        return True
                    return False

                def __call__(self, *args, **kwargs):
                    # If the model has been trained on the IPU: update the host side weights
                    self.copyWeightsToHostIfNeeded()
                    return parent.model.real_model_call(*args, **kwargs)

                def named_parameters(self, *args, **kwargs):
                    self.copyWeightsToHostIfNeeded()
                    return super().named_parameters(*args, **kwargs)

            # __call__ is an attribute, not a method, unfortunately we cannot just
            # replace it in the model object: we have to create a wrapper class
            # and change the object's class.
            m.__class__ = WrappedModel

    def _debugGetPopartIR(self):
        return poptorch_core._getPopartIR(self.executable)  # pylint: disable=protected-access

    # Copy weights from the device into the memory of the model given on wrapper creation.
    def copyWeightsToHost(self):
        poptorch_core.copyWeightsToHost_impl(  # pylint: disable=undefined-variable
            self.executable)

    # Write from host memory to IPU memory. This is done automatically on
    # compilation so should be rarely used.
    def copyWeightsToDevice(self):
        poptorch_core.copyWeightsToDevice_impl(  # pylint: disable=undefined-variable
            self.executable)

    def setOptimizer(self, optimizer):
        self.new_optimizer = optimizer

    def __call__(self, *args, **kwargs):
        # Convert single tensor to tuple.
        in_tensors = _Args(self.model, args, kwargs, self.training)

        if in_tensors.forEachMatchedAtLeastOnce(
                condition=lambda t: not t.is_contiguous(),
                doOnTrue=lambda t: t.contiguous()):
            if not self.warned_not_contiguous_input:
                logger.warning("At least one input tensor is not contiguous: "
                               "non-contiguous tensors will be converted.")
                self.warned_not_contiguous_input = True

        if self.executable is None:
            logger.info(
                "First time call to model will invoke poplar compilation."
                " %s %s", str(self.options.device_iterations),
                str(self.training))

            # Input will be in form of [BatchSize* BatchPerStep, ...] so we
            # should slice it up so we compile by the batch size alone.
            extra_poplar_batch_dims = self.options.device_iterations * \
                self.options.replication_factor * \
                self.options.Training.gradient_accumulation

            # There are two concepts of batch size. First is the "model" batch size then there is the
            # concept of batching at the popart level. Here we divide by the popart batch size so the
            # trace "sees" the model batch size but when we call execute we pass the full batch and popart
            # will partition it up.
            in_tensors_trace_view = _Args(self.model, args, kwargs,
                                          self.training)

            def narrowTensor(tensor):
                if not isinstance(tensor, torch.Tensor):
                    return tensor
                assert tensor.size()[0] % extra_poplar_batch_dims == 0, (
                    "Invalid batch dimension: In the input %s, the batch "
                    "dimension (%d) must be a multiple of "
                    "Options.deviceIterations(%d) * "
                    "Options.replicationFactor(%d) * "
                    "Options.Training.gradientAccumulation(%d) = %d "
                    "because it is used to calculate the batch size which will "
                    "be executed on the device in any given iteration. For a "
                    "full explanation see the batching semantics page of the "
                    "documentation.") % (
                        tensor.shape, tensor.size()[0],
                        self.options.device_iterations,
                        self.options.replication_factor,
                        self.options.Training.gradient_accumulation,
                        extra_poplar_batch_dims)
                return tensor.narrow(
                    0, 0,
                    tensor.size()[0] // extra_poplar_batch_dims)

            in_tensors_trace_view.forEach(narrowTensor)

            # Normal bools don't get captured in python.
            hasConvertedAnyHalf = [False]

            def possiblyConvertFromHalf(tensor):
                if tensor.dtype == torch.half:
                    hasConvertedAnyHalf[0] = True
                    return tensor.float()
                return tensor

            in_tensors_trace_view.forEach(possiblyConvertFromHalf)

            # Compile the poplar executable based on the batchsize.
            if self.options.Jit.trace_model:
                logger.info('Compiling the model using tracing')

                convertedLayers = []

                for name, layer in self.model.named_modules():
                    anyIsHalf = False
                    for param in layer.parameters():
                        if param.dtype == torch.half:
                            anyIsHalf = True
                            break

                    if anyIsHalf:
                        layer.float()

                        convertedLayers.append(name)

                # We will trace using the normal trace view.
                n = torch.jit.trace(self.model,
                                    in_tensors_trace_view.asTuple())

                # Convert any converted params back to half.
                for name, layer in n.named_modules():
                    if name in convertedLayers:
                        layer.half()

                if hasConvertedAnyHalf[0]:
                    # Get the originals back.
                    in_tensors_as_half = _Args(self.model, args, kwargs,
                                               self.training)
                    in_tensors_as_half.forEach(narrowTensor)

                    # Compile using the actual halves.
                    self.executable = poptorch_core.compileWithTrace(  # pylint: disable=undefined-variable
                        n._c, in_tensors_as_half.asTuple(),
                        self.options.toDict(), self.training, self.optimizer)
                else:
                    self.executable = poptorch_core.compileWithTrace(  # pylint: disable=undefined-variable
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

                self.executable = poptorch_core.compileWithScript(  # pylint: disable=undefined-variable
                    n._c, n.graph, in_tensors_trace_view.asTuple(),
                    self.options.toDict(), self.training)

        if self.options.connectionType == enums.ConnectionType.Never:
            logger.info(
                "Compilation complete and ConnectionType.Never selected:"
                " returning")
            return None

        # If this is an inference model: check if the same model is not being trained on a different IPU.
        # If it is: make sure the weights are updated.
        if not self.training:
            copyWeightsToHostIfNeeded = getattr(self.model,
                                                "copyWeightsToHostIfNeeded",
                                                None)
            if callable(copyWeightsToHostIfNeeded):
                if copyWeightsToHostIfNeeded():
                    # Weights have now been updated on the Host: copy them to the second IPU.
                    logger.debug("Implicit copyWeightsToDevice()")
                    self.copyWeightsToDevice()

        # Execute the poplar executable with the full size (batch * device interations)
        if self.new_optimizer and self.new_optimizer != self.optimizer:
            self.optimizer = self.new_optimizer
            output = poptorch_core.execute(  # pylint: disable=undefined-variable
                self.executable, in_tensors.asTuple(),
                convertOptimizerToDict(self.optimizer))
        else:
            output = poptorch_core.execute(  # pylint: disable=undefined-variable
                self.executable, in_tensors.asTuple(), {})

        if self.training:
            self.dirty_host_weights = True

        if len(output) > 1:
            return output
        return output[0]
