#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import transformers
import torch
import poptorch


def test_bert_small():
    torch.manual_seed(42)

    # Bert small.
    pretrained_weights = 'mrm8488/bert-small-finetuned-squadv2'
    model = transformers.BertModel.from_pretrained(pretrained_weights,
                                                   torchscript=True)
    tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_weights)

    # It *just* fits on one IPU but if the sequence length is too big it will need two.
    input_ids = torch.tensor([tokenizer.encode("E")])

    opts = poptorch.Options().profile(False)
    inference_model = poptorch.inferenceModel(model, opts)
    poptorchOut = inference_model(input_ids)

    native = model(input_ids)

    for poptorchResult, nativeResult in zip(poptorchOut, native):
        assert torch.allclose(poptorchResult,
                              nativeResult,
                              rtol=1e-02,
                              atol=1e-02)


def test_bert_medium_result():
    torch.manual_seed(42)

    # Bert small.
    pretrained_weights = 'mrm8488/bert-medium-finetuned-squadv2'
    model = transformers.BertForQuestionAnswering.from_pretrained(
        pretrained_weights)
    tokenizer = transformers.BertTokenizer.from_pretrained(
        pretrained_weights, return_token_type_ids=True)

    context = """Edinburgh is Scotland's compact, hilly capital."""
    question = "What is the capital of Scotland?"
    encoding = tokenizer.encode_plus(question, context)

    ins = encoding["input_ids"]
    input_ids = torch.tensor([encoding["input_ids"]])

    attention_mask = torch.tensor([encoding["attention_mask"]])
    start_scores_native, end_scores_native = model(
        input_ids, attention_mask=attention_mask)

    opts = poptorch.Options().profile(False)
    inference_model = poptorch.inferenceModel(model, opts)
    start_score_pop, end_scores_pop = inference_model(input_ids,
                                                      attention_mask)

    # Longer sequences begin to accumulate more floating point error.
    assert torch.allclose(start_scores_native,
                          start_score_pop,
                          rtol=1e-02,
                          atol=1e-02)
    assert torch.allclose(end_scores_native,
                          end_scores_pop,
                          rtol=1e-02,
                          atol=1e-02)

    assert torch.argmax(start_score_pop), torch.argmax(start_scores_native)
    assert torch.argmax(end_scores_pop), torch.argmax(end_scores_native)

    # Convert to string.
    ans_tokens = ins[torch.argmax(start_score_pop
                                  ):torch.argmax(end_scores_pop) + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens)

    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

    assert answer_tokens_to_string == 'edinburgh'
