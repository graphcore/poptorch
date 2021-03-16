# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import functools
import torch


class autocast:
    """ Creates an auto-casting region for the layers called inside this
        scope.

        >>> with poptorch.autocast():
        ...     layer()

        To turn off auto-casting for this region, set the keyword parameter
        explicitly.

        >>> with poptorch.autocast(enabled=False):
        ...     layer()
    """

    def __init__(self, enabled=True):
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            torch.ops.poptorch.begin_autocast()
        else:
            torch.ops.poptorch.suppress_autocast()

    def __exit__(self, type, value, traceback):
        torch.ops.poptorch.restore_autocast()

    def __call__(self, func):
        """ Function decorator for controlling whether autocast functionality is
            enabled. For example, turning autocasting on can be done as follows:

            >>> @poptorch.autocast()
            ... def forward(x):
            ...    ...

            To ensure that autocasting is off for a certain function, set the
            keyword parameter explicitly:

            >>> @poptorch.autocast(enabled=False)
            ... def forward(x):
            ...     ...
        """

        @functools.wraps(func)
        def autocastWrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return autocastWrapper


class Policy:
    """Base class for autocast policies. """

    def __init__(self, fp16=None, fp32=None, promote=None, demote=None):
        """Create a new automatic casting policy. Parameters are lists of torch
           operators or layers.

           :param list fp16: operators and layers to be cast to half precision
           :param list fp32: operators and layers to be cast to single precision
           :param list promote: operators and layers to be promoted to single
                                precision in a mixed-precision context
           :param list demote: operators and layers to be demoted to half
                               precision in a mixed-precision context
        """
        self.fp16 = fp16 or []
        self.fp32 = fp32 or []
        self.promote = promote or []
        self.demote = demote or []

    def applyToLayer(self, module, options):
        enabled = getattr(module, '_poptorch_autocast', False)
        if not enabled or not options.Precision.autocast_enabled:
            return

        module.forward = autocast()(module.forward)

        has_float = False
        has_half = False
        for p in module.parameters():
            has_half = has_half or (p.dtype == torch.half)
            has_float = has_float or (p.dtype == torch.float)

        mixed_precision = has_float and has_half

        if mixed_precision:
            if self._moduleInPolicyList(module, self.promote):
                module.float()
                return

            if self._moduleInPolicyList(module, self.demote):
                module.half()
                return

        if self._moduleInPolicyList(module, self.fp16):
            module.half()
            return

        if self._moduleInPolicyList(module, self.fp32):
            module.float()
            return

    def apply(self, module, options):
        # apply policy to all contained modules. Note that the first module
        # module is actually a self reference
        for layer in module.modules():
            self.applyToLayer(layer, options)

    def _isModuleClass(self, layer):
        mro = getattr(layer, '__mro__', None)
        if not mro:
            return False
        return torch.nn.Module in mro

    def _moduleInPolicyList(self, module, policy_list):
        for mod in policy_list:
            if self._isModuleClass(mod) and isinstance(module, mod):
                return True
        return False

    def _policyList(self, policy_list):
        result = []

        for op in policy_list:
            if self._isModuleClass(op):
                continue
            result.append(str(op).split(' ')[2])

        return result

    def _dict(self):
        policy = dict()
        policy['fp16'] = self._policyList(self.fp16)
        policy['fp32'] = self._policyList(self.fp32)
        policy['promote'] = self._policyList(self.promote)
        policy['demote'] = self._policyList(self.demote)
        return policy


def _autocast_attr(module, enabled=True):
    setattr(module, '_poptorch_autocast', enabled)


setattr(torch.nn.Module, 'autocast', _autocast_attr)

default_fp16 = [
    torch.addbmm, torch.addmm, torch.addmv, torch.addr, torch.baddbmm,
    torch.bmm, torch.chain_matmul, torch.conv1d, torch.conv2d, torch.conv3d,
    torch.conv_tbc, torch.conv_transpose1d, torch.conv_transpose2d,
    torch.conv_transpose3d, torch.convolution, torch.matmul, torch.mm,
    torch.mv, torch.nn.Conv2d
]

default_fp32 = []

default_promote = [
    torch.addcdiv, torch.addcmul, torch.atan2, torch.bilinear, torch.cat,
    torch.cross, torch.dot, torch.equal, torch.index_put, torch.stack,
    torch.tensordot
]

default_demote = []

# The default policy object
default = Policy(default_fp16, default_fp32, default_promote, default_demote)
