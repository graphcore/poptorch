# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import warnings
from typing import Any, Dict, List, Optional

try:
    import gcipuinfo  # type: ignore
except ImportError:
    gcipuinfo = None
import pandas as pd
import torch


def _get_clock_value() -> int:
    if gcipuinfo is None:
        default_clock_value = 1850
        warnings.warn('Unable to import gcipuinfo. Using default value '
                      f'{default_clock_value} MHz')
        return default_clock_value
    try:
        clock = int(gcipuinfo.gcipuinfo().getDevices()[0]['clock'][:-3])
    except Exception as e:
        raise RuntimeError(
            'Getting clock frequency using gcipuinfo failed') from e
    return clock


def _mean(col: pd.core.series.Series) -> Any:
    if pd.api.types.is_numeric_dtype(col):
        mean = col.mean()
        if col.name == 'cycles' or col.name == 'clock (MHz)':
            mean = mean.astype('int64')
        return mean

    return col.unique()


def to_data_frame(measurements: List[Dict[str, Any]],
                  calc_mean=False) -> pd.DataFrame:
    data_frame = pd.DataFrame(measurements)

    if calc_mean:
        return data_frame.agg(_mean)

    return data_frame


class PerfMetrics:
    r"""Track performance metrics from:
        * recorded number of cycles
        * sizes of input / output
    Defines an effective bandwidth from the size of the output result.
    """

    def __init__(self,
                 config_src: str,
                 operator: torch.nn.Module,
                 num_repeats: int,
                 op_name: str,
                 op_params: str,
                 clock: Optional[int] = None) -> None:
        output = operator.output
        numels = output.numel()
        numbytes = torch.finfo(output.dtype).bits // 8
        self.out_gib = numels * numbytes / 1024**3
        self.num_repeats = num_repeats
        self.clock = _get_clock_value() if clock is None else clock
        self.op_name = op_name
        self.op_params = op_params
        self.config_src = config_src

    def get_measurement(self, cycles: int) -> Dict[str, Any]:

        avg_cycles = cycles / self.num_repeats
        time_us = avg_cycles / self.clock
        time_s = time_us * 10**-6
        effective_bandwidth = self.out_gib / time_s

        return {
            'operator': self.op_name,
            'cycles': avg_cycles,
            'clock (MHz)': self.clock,
            'time (us)': time_us,
            'effective bandwidth (GiB/s)': effective_bandwidth,
            'parameters': self.op_params,
            'config source': self.config_src
        }
