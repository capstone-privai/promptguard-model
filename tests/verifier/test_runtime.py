"""The run-time verifier scores the detector's candidates in the order given, from the input the training records gave
the model for the same text, candidates and source. Tiny offline BERT and synthetic values only."""
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification

from privai_model.data.adapter import adapt_records
from privai_model.data.input_builder import build_input
from privai_model.inference.predict import ModelBundle
from privai_model.verifier import runtime
from privai_model.verifier.candidates import records_for_row

TEXT = '😀 setup\npassword = SynthPass_91\nbuild_id = Build_7731x\ndone'


def utf16(text, cp):
    return len(text[:cp].encode('utf-16-le')) // 2


def candidate(value, strength='ambiguous', score=None):
    start = TEXT.index(value)
    evidence = [dict(adapter='promptguard-rules', family='secret-assignment', strength='moderate')]
    if score is not None:
        evidence.append(dict(adapter='ml-span', family='ml', strength='ambiguous', score=score))
    return dict(start=utf16(TEXT, start), end=utf16(TEXT, start + len(value)), type='PASSWORD',
                rule_id='secret-assignment', detectors=['promptguard-rules'], strength=strength,
                legacy_protected=strength == 'strong', evidence=evidence)


@pytest.fixture
def verifier(tokenizer, monkeypatch):
    torch.manual_seed(0)
    model = BertForSequenceClassification(BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                                                     num_attention_heads=2, intermediate_size=32, num_labels=2))
    bundle = ModelBundle(tokenizer, model, torch.device('cpu'), 64, 'synthetic-verifier', 0.25)
    monkeypatch.setattr(runtime, 'load_model', lambda path, device, recall_target: bundle)
    return runtime.CandidateVerifier('unused', batch_size=1)


def test_scores_follow_the_candidates_and_the_bundle_threshold(verifier):
    candidates = [candidate('Build_7731x', score=0.4), candidate('SynthPass_91')]
    scores = verifier.score(TEXT, 'tool_output', candidates)
    assert len(scores) == 2 and all(0 <= s <= 1 for s in scores)
    assert verifier.score(TEXT, 'tool_output', candidates[::-1]) == pytest.approx(scores[::-1])
    assert verifier.score(TEXT, 'tool_output', []) == []
    assert (verifier.threshold, verifier.model_version) == (0.25, 'synthetic-verifier')


def test_model_input_matches_the_training_records(verifier, monkeypatch):
    seen = []
    real = runtime.predict_examples

    def capture(bundle, examples, batch_size):
        seen.extend(examples)
        return real(bundle, examples, batch_size)

    monkeypatch.setattr(runtime, 'predict_examples', capture)
    candidates = [candidate('SynthPass_91', 'strong'), candidate('Build_7731x', score=0.4)]
    verifier.score(TEXT, 'tool_output', candidates)
    row = dict(id='r1', text=TEXT, spans=[], source='tool_output', family_id='g1')
    trained = adapt_records(records_for_row(row, 'validation', candidates))
    assert [build_input(e).text for e in seen] == [build_input(e).text for e in trained]
    assert [(e.value, e.detector, e.type) for e in seen] == [(e.value, e.detector, e.type) for e in trained]


def test_rejects_bad_settings(verifier):
    with pytest.raises(ValueError):
        runtime.CandidateVerifier('unused', batch_size=0)
