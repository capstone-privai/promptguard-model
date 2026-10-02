import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification
from conftest import candidate
from privai_model.inference.predict import ModelBundle, predict_candidates


def test_inference_ids_and_reuse(tokenizer):
    model = BertForSequenceClassification(BertConfig(vocab_size=len(tokenizer), hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=2))
    bundle = ModelBundle(tokenizer,model,torch.device('cpu'),64,'synthetic-test',.5)
    records = [candidate(2),candidate(1)]
    for record in records:
        del record['label']
    rows = predict_candidates(bundle,records,batch_size=2)
    assert {r['candidate_id'] for r in rows} == {'synthetic:1','synthetic:2'}
    assert len(rows) == 2
    assert all(0<=r['p_mask']<=1 for r in rows)
    with pytest.raises(ValueError):
        predict_candidates(bundle,[records[0],records[0]])
    with pytest.raises(ValueError):
        predict_candidates(bundle,[{'granularity':'line'}])
