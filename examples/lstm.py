#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import poptorch


class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 3)

    def forward(self, input_tensors, hidden):
        Y, (Y_h, Y_c) = self.lstm(input_tensors, hidden)
        return Y, (Y_h, Y_c)


inputs = [torch.randn(1, 3) for _ in range(5)]
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state

inference_lstm = poptorch.inferenceModel(SimpleLSTM())
out, hidden = inference_lstm(inputs, hidden)

print(out)
print(hidden)
