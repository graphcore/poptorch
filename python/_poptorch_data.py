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
                 options: Optional['poptorch.Options'] = None,
                 training: Optional[bool] = None,
                 model: Optional['torch.nn.Module'] = None,
                 optimizer: Optional['torch.optim.Optimizer'] = None):
        self.options = options
        self.training = training
        self.model = model

        self.version = version
        self.optimizer = optimizer
        assert executable_inputs, "The executable's inputs are missing"
        self.executable_inputs = executable_inputs


def parse(filename: str, expected_version: str):
    """Extract the :py:class:`~poptorch.PoptorchData` and the offset at
       which the PopART executable is stored from a given file.
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
        assert data.version == expected_version, (
            "PopTorch version mismatch: "
            f"{filename} was created with version: {data.version}"
            f" and this is version {expected_version}")
        assert data.executable_inputs, (f"Invalid file {filename}:"
                                        " executable inputs are missing")
        if data.options:
            data.options._unfreeze()  # pylint: disable=protected-access
            # Remove usefOfflineIpuTarget related flags if used
            data.options.deleteIfExists("ipu_version")
            if data.options.connection_type == enums.ConnectionType.Never.value:
                data.options.connectionType(enums.ConnectionType.Always)

        return data, f.tell()
