#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import torch
import pytest
import poptorch

match_str = [
    """title: {
 {1.4962566 1.7682219}
 {1.0884774 1.1320305}
}""", """title: [
 [1.4962566e+00,
   1.7682219e+00]
 [1.0884774e+00,
   1.1320305e+00]
]""", """title: (
 (1.4962566;1.7682219)
 (1.0884774;1.1320305)
)"""
]

brackets = {
    "parentheses": ("(", ")"),
    "square": ("[", "]"),
    "curly": ("{", "}")
}


@pytest.mark.parametrize(
    "title,print_gradient,summarise_threshold,edge_items,"
    "max_line_width,digits,float_format,separator,brackets_type,"
    "match_str_idx",
    [("title", True, 1000, 3, 75, 8, "auto", None, "curly", 0),
     ("title", True, 500, 2, 15, 8, "scientific", ",", "square", 1),
     ("title", True, 1500, 1, 125, 8, "fixed", ";", "parentheses", 2)])
def test_print_ipu_tensor(capfd, title, print_gradient, summarise_threshold,
                          edge_items, max_line_width, digits, float_format,
                          separator, brackets_type, match_str_idx):
    separator = " " if separator is None else separator

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            x = x + 1
            x = poptorch.ipu_print_tensor(x, title, print_gradient,
                                          summarise_threshold, edge_items,
                                          max_line_width, digits, float_format,
                                          separator, *brackets[brackets_type])

            return x + self.bias

    poptorch_model = poptorch.inferenceModel(Model())

    torch.manual_seed(0)
    x = torch.rand((2, 2))

    _ = poptorch_model(x)

    captured = capfd.readouterr()

    # Very awkward to test this 'dynamically' so just test against some known
    # outputs above. Quite small tensors to test, but testing large ones would
    # be messy.

    assert match_str[match_str_idx] in captured.err
