import importlib.util
from pathlib import Path
from unittest.mock import patch
import json
import torch
import pytest
from transformers import BertConfig, BertModel, AutoModelForSequenceClassification
from privai_model.modeling.fusion import FeatureFusionClassifier
from privai_model.modeling.features import FeatureSpec, extract_features
from privai_model.modeling.model import load_checkpoint
from privai_model.inference.predict import ModelBundle, predict_candidates, load_model
from privai_model.data.synthetic import hard_cases
from privai_model.training.trainer import train


def small_fusion(tokenizer):
    return FeatureFusionClassifier(BertModel(BertConfig(vocab_size=len(tokenizer),hidden_size=16,
        intermediate_size=32,num_hidden_layers=1,num_attention_heads=2)),FeatureSpec())


def test_features_reach_classifier_and_encoder(tokenizer):
    torch.manual_seed(42)
    model=small_fusion(tokenizer).eval()
    ids=torch.tensor([[2,6,3]])
    mask=torch.ones_like(ids)
    f=torch.zeros((1,14),requires_grad=True)
    logits=model(ids,mask,f).logits
    logits[0,1].backward()
    assert f.grad.abs().sum()>0
    assert model.encoder.embeddings.word_embeddings.weight.grad.abs().sum()>0
    assert not torch.equal(logits, model(ids,mask,torch.ones((1,14))).logits)


def test_contract_ten_ids_reordering_and_no_loading(tokenizer):
    bundle=ModelBundle(tokenizer,small_fusion(tokenizer),torch.device('cpu'),64,'test',.5)
    records=hard_cases()[:10]
    for r in records: r.pop('label')
    with patch('privai_model.inference.predict.load_checkpoint',side_effect=AssertionError('unexpected reload')):
        a=predict_candidates(bundle,records,8)
        b=predict_candidates(bundle,list(reversed(records)),8)
        assert predict_candidates(bundle,[])==[]
    aa={r['candidate_id']:r['p_mask'] for r in a}
    bb={r['candidate_id']:r['p_mask'] for r in b}
    assert len(a)==len(b)==10 and aa.keys()==bb.keys()
    for cid in aa: assert aa[cid]==pytest.approx(bb[cid],abs=1e-6)


def test_fusion_roundtrip_and_metadata(tokenizer,tmp_path):
    model=small_fusion(tokenizer).eval()
    before=predict_candidates(ModelBundle(tokenizer,model,torch.device('cpu'),64,'test',.5),hard_cases()[:2])
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    meta={'model_version':'test','window_size':64,'thresholds':{'0.99':.5},'deployment_recall_target':.99,
          'character_features':True,'feature_spec':model.feature_spec.to_dict()}
    (tmp_path/'bundle.json').write_text(json.dumps(meta))
    after=predict_candidates(load_model(tmp_path,'cpu'),hard_cases()[:2])
    assert [r['p_mask'] for r in before]==[r['p_mask'] for r in after]
    meta['feature_spec']['prefixes'].reverse()
    (tmp_path/'bundle.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError): load_model(tmp_path,'cpu')


def test_feature_train_save_reload(tmp_path):
    spec=importlib.util.spec_from_file_location('smoke',Path(__file__).parents[1]/'scripts/smoke.py')
    smoke=importlib.util.module_from_spec(spec); spec.loader.exec_module(smoke)
    config=smoke.prepare(tmp_path/'source')
    config.update(character_features=True,evaluate_test=False)
    threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        result=train(config)
        assert result['test'] is None
        bundle=load_model(Path(config['output_dir'])/'model','cpu')
        assert bundle.model.feature_spec.dimension==14
        assert len(predict_candidates(bundle,hard_cases()[:10],8))==10
        meta=json.loads((Path(config['output_dir'])/'model/bundle.json').read_text())
        assert meta['feature_dimension']==14 and meta['threshold_status']=='synthetic_only'
        final_spec=importlib.util.spec_from_file_location('final_eval',Path(__file__).parents[1]/'scripts/final_evaluate.py')
        final=importlib.util.module_from_spec(final_spec);final_spec.loader.exec_module(final)
        with patch('privai_model.evaluation.evaluator.choose_threshold',side_effect=AssertionError('test must not retune')):
            report=final.finalize(config['data_path'],config['output_dir'],tmp_path/'final','cpu',4)
        assert report['thresholds']==meta['thresholds']
        assert report['test']['0.99']['records']==4
    finally: torch.set_num_threads(threads)
