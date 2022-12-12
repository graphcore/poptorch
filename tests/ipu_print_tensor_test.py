#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import torch
import pytest
import poptorch


match_str_0 = \
"""title: {
 {1.4962566 1.7682219}
 {1.0884774 1.1320305}
}"""

match_str_1 = \
"""title: [
 [1.4962566e+00,
   1.7682219e+00]
 [1.0884774e+00,
   1.1320305e+00]
]"""

match_str_2 = \
"""title: (
 (1.4962566;1.7682219)
 (1.0884774;1.1320305)
)"""


@pytest.mark.parametrize(
    "title,print_gradient,summarise_threshold,edge_items,"
    "max_line_width,digits,float_format,separator,open_bracket,"
    "close_bracket,match_str",
    [("title", True, 1000, 3, 75, 8, "auto", " ", "{", "}", match_str_0),
     ("title", True, 500, 2, 15, 8, "scientific", ",", "[", "]", match_str_1),
     ("title", True, 1500, 1, 125, 8, "fixed", ";", "(", ")", match_str_2)])
def test_print_ipu_tensor(capfd, title, print_gradient, summarise_threshold,
                          edge_items, max_line_width, digits, float_format,
                          separator, open_bracket, close_bracket, match_str):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            x = x + 1
            x = poptorch.ipu_print_tensor(x, title, print_gradient,
                                          summarise_threshold, edge_items,
                                          max_line_width, digits, float_format,
                                          separator, open_bracket,
                                          close_bracket)

            return x + self.bias

    poptorch_model = poptorch.inferenceModel(Model())

    torch.manual_seed(0)
    x = torch.rand((2, 2))

    _ = poptorch_model(x)

    captured = capfd.readouterr()

    # Very awkward to test this 'dynamically' so just test against some known
    # outputs above. Quite small tensors to test, but testing large ones would
    # be messy.
    assert match_str in captured.err
