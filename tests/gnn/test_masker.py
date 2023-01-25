# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import torch
import torch_geometric as pyg

from poptorch_geometric import masker


@pytest.fixture(params=[True, False])
def entries(request) -> masker.Entries:
    """Returns something which looks like an entry"""
    pyg.seed_everything(1)
    is_tuple = request.param
    entry = torch.rand([2, 3, 4])
    return (entry, entry) if is_tuple else entry


class TestNoOpMasker:
    """Tests the No Op masker, makes sure it does nothing."""

    @pytest.mark.parametrize("masker_name", ["node", "graph", "edge"])
    def test_masker_does_not_change_the_object(self, masker_name: str,
                                               entries: masker.Entries):
        mask = masker.NoMasker()
        output_entries = getattr(mask, f"{masker_name}_masker")(entries)
        assert entries is output_entries


class TestNoOpLayerMasker:
    @pytest.fixture
    def layer(self):
        def layer_function(*args):
            total = 0
            for arg in args:
                total += torch.sum(arg)
            return total

        return layer_function

    @pytest.mark.parametrize("masker_name", ["node", "graph", "edge"])
    def test_masker_does_not_change_the_layer_result(
            self,
            masker_name: str,
            entries: masker.Entries,
            layer: masker.Layer,
    ):
        mask = masker.PreLayerMasker(masker=masker.NoMasker())
        masked_layer = getattr(mask, f"{masker_name}_masker")(layer)
        if not isinstance(entries, (tuple, list)):
            entries = (entries, )
        reference_output = layer(*entries)
        masked_output = masked_layer(*entries)
        assert reference_output == masked_output, (
            "For the No-op layer masker," +
            " the result of a layer should be unchanged")
