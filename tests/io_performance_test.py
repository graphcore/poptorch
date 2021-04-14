#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os  # pylint: disable=unused-import
import unittest.mock
import numpy as np
import pytest
import torch
import poptorch
import helpers

IMAGE_SIZE = (3, 512, 512)
DATASET_SIZE = 1000
BATCH_SIZE = 16


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, io_dtype):
        super().__init__()
        self.io_dtype = io_dtype

    def __len__(self):
        return DATASET_SIZE

    def __getitem__(self, _):
        return torch.randint(0, 256, IMAGE_SIZE).to(self.io_dtype)


def get_mean_cycle_count(io_dtype, capfd):
    class Model(torch.nn.Module):
        def forward(self, x):
            x = x.to(torch.float32)
            x = x * 2
            return x.to(io_dtype)

    opts = poptorch.Options()
    opts.logCycleCount(True)
    data_loader = poptorch.DataLoader(
        opts,
        ImageDataset(io_dtype),
        BATCH_SIZE,
        shuffle=False,
        drop_last=True,
    )
    model = poptorch.inferenceModel(Model(), opts)

    num_iterations = 0
    for x in data_loader:
        num_iterations += 1
        _ = model(x)
    data_loader.terminate()

    log_matches = helpers.LogChecker(capfd).createIterator().findAll(
        r'Total number of IPU cycles: (\d+)')
    assert len(log_matches) == num_iterations

    cycle_counts = []
    for match in log_matches:
        cycle_counts.append(int(match.group(1)))
    return np.array(cycle_counts).mean()


@pytest.mark.skipif(not poptorch.ipuHardwareIsAvailable(),
                    reason="Hardware IPU needed")
@pytest.mark.parametrize("io_dtype1,io_dtype2,min_dtype1_div_dtype2_ratio",
                         [(torch.float32, torch.int8, 2.6),
                          (torch.float32, torch.uint8, 2.6),
                          (torch.float32, torch.float16, 1.2)])
@helpers.printCapfdOnExit
@unittest.mock.patch.dict("os.environ", helpers.disableAllModels())
def test_compare_io_performance(capfd, io_dtype1, io_dtype2,
                                min_dtype1_div_dtype2_ratio):
    poptorch.setLogLevel('INFO')
    cycle_count_1 = get_mean_cycle_count(io_dtype1, capfd)
    cycle_count_2 = get_mean_cycle_count(io_dtype2, capfd)

    # We allow quite large differences as performance has large variance between
    # the runs. This test should thus only indicate serious regressions.
    assert cycle_count_1 / cycle_count_2 >= min_dtype1_div_dtype2_ratio
