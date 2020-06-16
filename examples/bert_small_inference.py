#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import transformers
import torch
import poptorch

pretrained_weights = 'mrm8488/bert-small-finetuned-squadv2'
model = transformers.BertModel.from_pretrained(pretrained_weights,
                                               torchscript=True)

tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_weights)
input_ids = torch.tensor([tokenizer.encode("Example text")])

# Mark pipeline
model.encoder.layer[1] = poptorch.IPU(1, model.encoder.layer[1])

inference_model = poptorch.inferenceModel(model, profile=True)
out = inference_model(input_ids)
