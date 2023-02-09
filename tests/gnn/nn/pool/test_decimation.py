# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch

from torch_geometric.nn.pool.decimation import decimation_indices

from pool_utils import op_harness


def test_decimation_basic(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: 'NotImplementedError: Could not run "
        "'aten::_local_scalar_dense' with arguments from the 'Meta' backend'. "
        "Will be enabled after AFS-144 is fixed.")

    N_1, N_2 = 4, 6
    decimation_factor = 2
    ptr = torch.tensor([0, N_1, N_1 + N_2])

    idx_decim, ptr_decim = op_harness(decimation_indices,
                                      [ptr, decimation_factor],
                                      assert_func=lambda x, y: True)

    expected_size = (N_1 // decimation_factor) + (N_2 // decimation_factor)
    assert idx_decim.size(0) == expected_size

    expected = torch.tensor([0, N_1 // decimation_factor, expected_size])
    assert torch.equal(ptr_decim, expected)


def test_decimation_single_cloud(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: 'NotImplementedError: Could "
        "not run 'aten::_local_scalar_dense' with arguments from the 'Meta' "
        "backend.'. Will be enabled after AFS-144 is fixed.")

    N_1 = 4
    decimation_factor = 2
    ptr = torch.tensor([0, N_1])

    idx_decim, ptr_decim = op_harness(decimation_indices,
                                      [ptr, decimation_factor])

    expected_size = N_1 // decimation_factor
    assert idx_decim.size(0) == expected_size
    assert torch.equal(ptr_decim, torch.tensor([0, expected_size]))


def test_decimation_almost_empty(request):
    pytest.skip(
        f"{request.node.nodeid}: Error: 'NotImplementedError: Could "
        "not run 'aten::_local_scalar_dense' with arguments from the 'Meta' "
        "backend.'. Will be enabled after AFS-144 is fixed.")

    N_1 = 4
    decimation_factor = 666  # greater than N_1
    ptr = torch.tensor([0, N_1])

    idx_decim, ptr_decim = op_harness(decimation_indices,
                                      [ptr, decimation_factor])

    assert idx_decim.size(0) == 1
    assert torch.equal(ptr_decim, torch.tensor([0, 1]))
