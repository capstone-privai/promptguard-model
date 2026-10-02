import torch
import pytest
from transformers import BertConfig,BertForSequenceClassification
from privai_model.data.adapter import adapt_records
from privai_model.data.synthetic import hard_cases
from privai_model.inference.predict import ModelBundle
from privai_model.benchmarking import benchmark,device_status


def test_benchmark_full_windows_and_statistics(tokenizer):
    model=BertForSequenceClassification(BertConfig(vocab_size=len(tokenizer),hidden_size=8,
        intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=1024))
    bundle=ModelBundle(tokenizer,model,torch.device('cpu'),64,'synthetic',.5)
    threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        result=benchmark(bundle,adapt_records(hard_cases(True)),windows=[64,256,512,1024],batch_sizes=[1,8],warmup=1,repeats=2,require_full_windows=True)
        assert len(result['results'])==8
        for row in result['results']:
            assert all(n==row['window_size'] for n in row['token_lengths'])
            assert row['mean_latency_ms']>0 and row['p95_latency_ms']>=row['median_latency_ms']
            assert row['throughput_candidates_per_second']>0
        assert bundle.window_size==64
        with pytest.raises(ValueError): benchmark(bundle,adapt_records(hard_cases()[:1]),batch_sizes=[8])
    finally: torch.set_num_threads(threads)


def test_device_report():
    status=device_status()
    assert status['cpu_ok']
    assert isinstance(status['mps_available'],bool)
