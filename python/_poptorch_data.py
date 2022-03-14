# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import pickle
from typing import Any, List, Optional

# Do not import any poptorch.* here: it will break the poptorch module
from . import enums


class PoptorchData:
    """Metadata to save when exporting an executable in order to be able
    to reload it.

    Note: :py:func:`poptorch.load` can only be used if all the arguments are
    provided
    :py:meth:`poptorch.PoplarExecutor.loadExecutable` can be used in either
    case (But only version and executable_inputs will be used)
    """

    def __init__(self,
                 version: str,
                 executable_inputs: List[Any],
                 options: 'poptorch.Options',
                 training: Optional[bool] = None,
                 model: Optional['torch.nn.Module'] = None,
                 optimizer: Optional['torch.optim.Optimizer'] = None,
                 random_seed: Optional[int] = None,
                 rng_state: Optional[List[int]] = None):
        self.options = options
        self.training = training
        self.model = model

        self.version = version
        self.optimizer = optimizer
        assert executable_inputs, "The executable's inputs are missing"
        self.executable_inputs = executable_inputs
        self.random_seed = random_seed
        self.rng_state = rng_state

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
        if opt is None:
            self.optimizer_state = None
        else:
            self.optimizer_state = opt.state_dict()


def parse(serialized_data: bytes, expected_version: str):
    """Extract the :py:class:`~poptorch.PoptorchData` and the offset at
       which the PopART executable is stored from a given file.
    """
    data = pickle.loads(serialized_data)
    assert data.version == expected_version, (
        "PopTorch version mismatch: "
        f"File was created with version: {data.version}"
        f" and this is version {expected_version}")
    assert data.executable_inputs, ("Executable inputs are missing")

    if data.options:
        data.options._unfreeze()  # pylint: disable=protected-access
        # Remove usefOfflineIpuTarget related flags if used
        data.options.deleteIfExists("ipu_version")
        if data.options.connection_type == enums.ConnectionType.Never.value:
            data.options.connectionType(enums.ConnectionType.Always)

    return data
