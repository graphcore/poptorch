# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import transformers
import torch
import poptorch

# A bert model from hugging face. See the packaged BERT example for actual usage.
model = transformers.BertForQuestionAnswering.from_pretrained(
    'mrm8488/bert-medium-finetuned-squadv2')


# A handy way of seeing the names of all the layers in the network.
print(model)

# All layers before "model.bert.encoder.layer[0]" will be on IPU 0 and all layers from
# "model.bert.encoder.layer[0]" onwards (inclusive) will be on IPU 1.
model.bert.encoder.layer[0] = poptorch.IPU(
    1, model.bert.encoder.layer[0])

# Now all layers before layer are on IPU 1 and this layer onward is on IPU 2
model.bert.encoder.layer[2] = poptorch.IPU(
    2, model.bert.encoder.layer[2])

# Finally all layers from this layer till the end of the network are on IPU 3.
model.bert.encoder.layer[4] = poptorch.IPU(
    3, model.bert.encoder.layer[4])

# We must batch the data by at least the number of IPUs. Each IPU will still execute
# whatever the model batch size is.
data_batch_size = 4

# Model is now passed to the wrapper as usual.
inference_model = poptorch.inferenceModel(model, device_iterations=data_batch_size)



class Network(nn.Module):
    def forward(self, x):

        # Implicitly layers are on IPU 0 until a with IPU annotation is encountered.
        x = self.layer1(x)
        with poptorch.IPU(1):
            x = self.layer2(x)
            x = x.view(-1, 320)

        with poptorch.IPU(2):
            x = self.layer3_act(self.layer3(x))
            x = self.layer4(x)

        with poptorch.IPU(3):
            x = self.softmax(x)
        return x
