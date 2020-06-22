#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import transformers
import torch
import poptorch

tokenizer = transformers.BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad',
    return_token_type_ids=True)
model = transformers.BertForQuestionAnswering.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')

context = """Graphcore is a Bristol based semiconductor company that develops accelerators for AI and machine learning.
            It aims to make a massively parallel Intelligence Processing Unit (IPU) that holds the complete
            machine learning model inside the processor. Graphcore was founded in 2016 by CTO Simon Knowles and CEO Nigel Toon"""

questions = [
    "What is graphcore doing?", "Who founded Graphcore?",
    "Where is Graphcore's office?", "What position does Nigel have?",
    "What does the acronym IPU stand for?", "How does the IPU work?"
]

inference_model = poptorch.inferenceModel(model)

print("Context: " + context)
for question in questions:
    # Encode the query and context.
    encoding = tokenizer.encode_plus(question,
                                     context,
                                     max_length=80,
                                     pad_to_max_length='right')

    input_ids, attention_mask = encoding["input_ids"], encoding[
        "attention_mask"]

    # Execute on IPU.
    start_score_pop, end_scores_pop = inference_model(
        torch.tensor([input_ids]), torch.tensor([attention_mask]))

    # Convert the token back to the string.
    answer_ids = input_ids[torch.argmax(start_score_pop
                                        ):torch.argmax(end_scores_pop) + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids,
                                                    skip_special_tokens=True)
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    print("Query: " + question)
    print("Answer: " + answer)
