import transformers
import torch
import poptorch


def test_bert_small():

    # Bert small.
    pretrained_weights = 'mrm8488/bert-small-finetuned-squadv2'
    model = transformers.BertModel.from_pretrained(pretrained_weights,
                                                   torchscript=True)
    tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_weights)
    input_ids = torch.tensor([tokenizer.encode("Example text")])

    # Compile and run the bert model.
    inference_model = poptorch.inferenceModel(model, profile=False)
    out = inference_model(input_ids)

    # We are just checking that it compiles and runs. Checking the results is still a TODO
    assert True
