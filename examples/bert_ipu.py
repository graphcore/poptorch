#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import transformers
import torch
import poptorch

tokenizer = transformers.BertTokenizer.from_pretrained(
    'mrm8488/bert-medium-finetuned-squadv2', return_token_type_ids=True)
model = transformers.BertForQuestionAnswering.from_pretrained(
    'mrm8488/bert-medium-finetuned-squadv2')

context = """Scotland is a country that is part of the United Kingdom. Covering the northern third of
            the island of Great Britain, mainland Scotland has a 96 mile (154 km) border with England
            to the southeast and is otherwise surrounded by the Atlantic Ocean to the north and west,
            the North Sea to the northeast and the Irish Sea to the south. In addition, Scotland includes
            more than 790 islands; principally within the Northern Isles and the Hebrides archipelagos."""

questions = [
    "How many islands are there in Scotland?",
    "What sea is to the south of Scotland",
    "Where is England in relation to Scotland?",
    "How long is the border between England and Scotland?"
]

batches = len(questions)

running_on_hardware = False

# Pipeline the model over two IPUs. You must have at least as many batches (questions) as you have IPUs.
if running_on_hardware:
    model.bert.embeddings.position_embeddings = poptorch.IPU(
        1, model.bert.embeddings.position_embeddings)

# Mark model for inference.
inference_model = poptorch.inferenceModel(model, device_iterations=batches)

# Batch by the number of iterations so we fill the pipeline.
encoding, input_ids, attention_mask = [None] * batches, [None] * batches, [
    None
] * batches

# Encode the query and context.
batch_list, atten_list = [], []

# Encode each question for the IPU.
for i in range(0, batches):
    encoding[i] = tokenizer.encode_plus(questions[i],
                                        context,
                                        max_length=110,
                                        pad_to_max_length='right')
    input_ids[i], attention_mask[i] = encoding[i]["input_ids"], encoding[i][
        "attention_mask"]
    batch_list.append(input_ids[i])
    atten_list.append(attention_mask[i])

input_batch = torch.tensor(batch_list)
attention_batch = torch.tensor(atten_list)

print(input_batch.size())
# Execute on IPU.
start_score_pop, end_scores_pop = inference_model(input_batch, attention_batch)

print("Context: " + context)
index = 0
for start_score, end_score in zip(start_score_pop, end_scores_pop):
    answer_ids = input_ids[index][torch.argmax(start_score
                                               ):torch.argmax(end_score) + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids,
                                                    skip_special_tokens=True)
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    print("Question : " + questions[index])
    print("Answer : " + answer)

    index += 1
