# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional

import torch

import poptorch


class BenchModel(torch.nn.Module):
    def __init__(self, operator: torch.nn.Module, num_repeats: int) -> None:
        super().__init__()
        self.num_repeats = num_repeats
        self.operator = operator

    def forward(self) -> torch.Tensor:
        return poptorch.for_loop(self.num_repeats, self.operator,
                                 self.operator.loop_inputs())[-1]


def _create_poptorch_options(
        synthetic_data: bool = 0,
        available_memory_proportion: Optional[float] = None,
        profile_dir: Optional[str] = None,
        cache_dir: str = 'benchgnn_model_cache') -> poptorch.Options:
    options = poptorch.Options()
    options.enableSyntheticData(synthetic_data)
    options.logCycleCount(True)
    options.enableExecutableCaching(cache_dir)
    options.connectionType(poptorch.ConnectionType.OnDemand)

    if available_memory_proportion is not None:
        amp_dict = {"IPU0": available_memory_proportion}
        options.setAvailableMemoryProportion(amp_dict)

    if profile_dir:
        options.enableProfiling(profile_dir)
    return options


class BenchModelBuilder():
    def __init__(self,
                 synthetic_data: bool = False,
                 available_memory_proportion: Optional[float] = None,
                 profile_dir: Optional[str] = None,
                 cache_dir: str = 'benchgnn_model_cache',
                 num_repeats: int = 128) -> None:
        """
        model compile options

        Args:
            synthetic_data (bool, optional): Use synthetic data on the device
                to disable I/O. (default: :obj:`False`)
            available_memory_proportion (float, optional): the AMP budget
                used for planning ops. (default: :obj:`None`)
            profile_dir (str, optional): saves the profiling report to the
                provided location. (default: :obj:`None`)
            cache_dir (str, optional): saves the executable cache to the
                provided location. (default: :obj:`benchgnn_model_cache`)
            num_repeats (int, optional): the number of times to invoke the
                operator on device. (default: :obj:`128`)
        """
        self.num_repeats = num_repeats
        self.options = _create_poptorch_options(synthetic_data,
                                                available_memory_proportion,
                                                profile_dir, cache_dir)

    def create_model(self, operator: torch.nn.Module):
        model = BenchModel(num_repeats=self.num_repeats, operator=operator)
        pop_model = poptorch.inferenceModel(model, options=self.options)
        pop_model.compile()
        return pop_model
