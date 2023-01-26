# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
from conv_utils import get_dataset


@pytest.fixture
def dataset():
    return get_dataset()
