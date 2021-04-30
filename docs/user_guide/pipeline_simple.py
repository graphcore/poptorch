# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
# annotations_start
import transformers
import torch
import poptorch

# A bert model from hugging face. See the packaged BERT example for actual usage.
pretrained_weights = 'mrm8488/bert-medium-finetuned-squadv2'


# For later versions of transformers, we need to wrap the model and set
# return_dict to False
class WrappedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wrapped = transformers.BertForQuestionAnswering.from_pretrained(
            pretrained_weights)

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.wrapped.forward(input_ids,
                                    attention_mask,
                                    token_type_ids,
                                    return_dict=False)

    def __getattr__(self, attr):
        try:
            return torch.nn.Module.__getattr__(self, attr)
        except torch.nn.modules.module.ModuleAttributeError:
            return getattr(self.wrapped, attr)


model = WrappedModel()

# A handy way of seeing the names of all the layers in the network.
print(model)

# All layers before "model.bert.encoder.layer[0]" will be on IPU 0 and all layers from
# "model.bert.encoder.layer[0]" onwards (inclusive) will be on IPU 1.
model.bert.encoder.layer[0] = poptorch.BeginBlock(model.bert.encoder.layer[0],
                                                  ipu_id=1)

# Now all layers before layer are on IPU 1 and this layer onward is on IPU 2
model.bert.encoder.layer[2] = poptorch.BeginBlock(model.bert.encoder.layer[2],
                                                  ipu_id=2)

# Finally all layers from this layer till the end of the network are on IPU 3.
model.bert.encoder.layer[4] = poptorch.BeginBlock(model.bert.encoder.layer[4],
                                                  ipu_id=3)

# We must batch the data by at least the number of IPUs. Each IPU will still execute
# whatever the model batch size is.
data_batch_size = 4

# Create a poptorch.Options instance to override default options
opts = poptorch.Options()
opts.deviceIterations(data_batch_size)
# annotations_end

# Model is now passed to the wrapper as usual.
inference_model = poptorch.inferenceModel(model, opts)

tokenizer = transformers.BertTokenizer.from_pretrained(
    "mrm8488/bert-medium-finetuned-squadv2", return_token_type_ids=True)

# Make use of the model
contexts = [
    """Edinburgh is Scotland's compact, hilly capital.""",
    """The oldest cat recorded was Cream Puff at 38 years.""",
    """The largest litter of kittens produced 19 kittens.""",
    """The first webcam was used to check the status of a coffee pot."""
]
questions = [
    "What is the capital of Scotland?", "How old was the oldest cat ever?",
    "How many kittens in the largest litter?",
    "What was the first webcam used for?"
]

encoding = tokenizer(questions, contexts, padding=True)

input_ids = encoding["input_ids"]

start_scores, end_scores = inference_model(
    torch.tensor(encoding["input_ids"]),
    torch.tensor(encoding["attention_mask"]),
    torch.tensor(encoding["token_type_ids"]))

answer_string = []
for batch_id in range(len(contexts)):
    ans_tokens = input_ids[batch_id][torch.argmax(start_scores[batch_id]):torch
                                     .argmax(end_scores[batch_id]) + 1]

    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens)
    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    answer_string.append(answer_tokens_to_string)

print(answer_string)
assert answer_string[0] == 'edinburgh'
assert answer_string[1] == '38 years'
assert answer_string[2] == '19'
assert answer_string[3] == 'to check the status of a coffee pot'


# annotations_inline_start
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(5, 10)
        self.layer2 = torch.nn.Linear(10, 5)
        self.layer3 = torch.nn.Linear(5, 5)
        self.layer4 = torch.nn.Linear(5, 5)

        self.act = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):

        # Explicit layers on a certain IPU
        poptorch.Block.useAutoId()
        with poptorch.Block(ipu_id=0):
            x = self.act(self.layer1(x))

        with poptorch.Block(ipu_id=1):
            x = self.act(self.layer2(x))

        with poptorch.Block(ipu_id=2):
            x = self.act(self.layer3(x))
            x = self.act(self.layer4(x))

        with poptorch.Block(ipu_id=3):
            x = self.softmax(x)
        return x


model = Network()
opts = poptorch.Options()
opts.deviceIterations(4)
poptorch_model = poptorch.inferenceModel(model, options=opts)
print(poptorch_model(torch.rand((4, 5))))

# annotations_inline_end


# pylint: disable=function-redefined
# annotations_decorator_start
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(5, 10)
        self.layer2 = torch.nn.Linear(10, 5)
        self.layer3 = torch.nn.Linear(5, 5)
        self.layer4 = torch.nn.Linear(5, 5)

        self.act = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        poptorch.Block.useAutoId()
        x = self.block_one(x)
        x = self.block_two(x)
        x = self.final_activation(x)
        return x

    @poptorch.BlockFunction(ipu_id=0)
    def block_one(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        return x

    @poptorch.BlockFunction(ipu_id=1)
    def block_two(self, x):
        x = self.act(self.layer3(x))
        x = self.act(self.layer4(x))
        return x

    @poptorch.BlockFunction(ipu_id=1)
    def final_activation(self, x):
        return self.softmax(x)


model = Network()
opts = poptorch.Options()
opts.deviceIterations(4)
poptorch_model = poptorch.inferenceModel(model, options=opts)
print(poptorch_model(torch.rand((4, 5))))
# annotations_decorator_end
