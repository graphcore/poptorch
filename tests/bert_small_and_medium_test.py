#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import transformers
import torch
import helpers
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

    options = poptorch.Options()
    inference_model = poptorch.inferenceModel(model, options)
    poptorch_out = inference_model(input_ids)

    native = model(input_ids)

    for poptorchResult, nativeResult in zip(poptorch_out, native):
        helpers.assert_allclose(actual=poptorchResult,
                                expected=nativeResult,
                                rtol=1e-02,
                                atol=1e-02)


def test_bert_small_half():
    torch.manual_seed(42)

    # Bert small.
    pretrained_weights = 'mrm8488/bert-small-finetuned-squadv2'
    model = transformers.BertModel.from_pretrained(pretrained_weights,
                                                   torchscript=True)
    tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_weights)

    # It *just* fits on one IPU but if the sequence length is too big it will need two.
    input_ids = torch.tensor([tokenizer.encode("E")])

    model.half()
    options = poptorch.Options()
    inference_model = poptorch.inferenceModel(model, options)
    poptorch_out = inference_model(input_ids)

    # Just check that we compile for now.
    assert poptorch_out[0].dtype == torch.half


def test_bert_medium_result():
    torch.manual_seed(42)

    pretrained_weights = 'mrm8488/bert-medium-finetuned-squadv2'

    # For later versions of transformers, we need to wrap the model and set
    # return_dict to False
    class WrappedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            transformers_BFQA = transformers.BertForQuestionAnswering
            self.wrapped = transformers_BFQA.from_pretrained(
                'mrm8488/bert-medium-finetuned-squadv2')

        def forward(self, input_ids, attention_mask):
            return self.wrapped.forward(input_ids,
                                        attention_mask,
                                        return_dict=False)

        def __getattr__(self, attr):
            try:
                return torch.nn.Module.__getattr__(self, attr)
            except AttributeError:
                return getattr(self.wrapped, attr)

    model = WrappedModel()

    tokenizer = transformers.BertTokenizer.from_pretrained(
        pretrained_weights, return_token_type_ids=True)

    context = """Edinburgh is Scotland's compact, hilly capital."""
    question = "What is the capital of Scotland?"
    encoding = tokenizer.encode_plus(question, context)

    mask = encoding["attention_mask"]
    ins = encoding["input_ids"]
    input_ids = torch.tensor([ins, ins])

    attention_mask = torch.tensor([mask, mask])
    start_scores_native, end_scores_native = model(
        input_ids, attention_mask=attention_mask)

    opts = poptorch.Options()
    opts.deviceIterations(2)

    model.bert.embeddings.position_embeddings = poptorch.BeginBlock(
        model.bert.embeddings.position_embeddings, ipu_id=1)

    inference_model = poptorch.inferenceModel(model, opts)
    start_score_pop, end_scores_pop = inference_model(input_ids,
                                                      attention_mask)

    # Longer sequences begin to accumulate more floating point error.
    helpers.assert_allclose(expected=start_scores_native,
                            actual=start_score_pop,
                            rtol=1e-02,
                            atol=1e-02)
    helpers.assert_allclose(expected=end_scores_native,
                            actual=end_scores_pop,
                            rtol=1e-02,
                            atol=1e-02)

    assert torch.argmax(start_score_pop), torch.argmax(start_scores_native)
    assert torch.argmax(end_scores_pop), torch.argmax(end_scores_native)

    # Convert to string (Only check the first result as we've already established the two were identical)
    ans_tokens = ins[torch.argmax(start_score_pop[0]
                                  ):torch.argmax(end_scores_pop[0]) + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens)

    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

    assert answer_tokens_to_string == 'edinburgh'
