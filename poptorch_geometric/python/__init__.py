# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import importlib

from .collate import make_exclude_keys
from .common import call_once
from .dataloader import (FixedSizeDataLoader, FixedSizeStrategy,
                         OverSizeStrategy)
from .fixed_size_options import FixedSizeOptions
from .types import PyGArgsParser, registerCustomArgParsers
from .utils import TrainingStepper, set_aggregation_dim_size
from .override import _TorchGeometricOpsSubstitutionManager

__version__ = "@VERSION@-@SNAPSHOT@"

__all__ = [
    '__version__', 'FixedSizeDataLoader', 'FixedSizeOptions',
    'FixedSizeStrategy', 'set_aggregation_dim_size', 'TrainingStepper',
    'make_exclude_keys', 'OverSizeStrategy', 'PyGArgsParser'
]


@call_once
def registerOverrideManager():
    poplar_executor_spec = importlib.util.find_spec(
        "poptorch._poplar_executor")
    if poplar_executor_spec is not None:
        loader = poplar_executor_spec.loader
        if loader is not None:
            poplar_executor = loader.load_module()
            poplar_executor._OverwriteContextManager.registerSubsitutionManager(  # pylint: disable=protected-access
                _TorchGeometricOpsSubstitutionManager)


registerOverrideManager()
registerCustomArgParsers()
