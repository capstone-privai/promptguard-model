import json
from pathlib import Path
from privai_model.ml_span.function_words import POLICY
from privai_model.ml_span.short_values import PROMPTGUARD_MIN_VALUE_LENGTH
from privai_model.ml_span.schema import load
from privai_model.ml_span.splitter import assert_disjoint

ROOT = Path(__file__).resolve().parents[2]
V10 = ROOT / 'models/span-detector-v10'


def read(path):
    return json.loads((ROOT / path).read_text())


def test_v10_bundle_uses_validation_cutoff_and_runtime_filter():
    meta = read('models/span-detector-v10/span_bundle.json')
    calibration = read('reports/ml_span/context-v10-warm-exact-calibration.json')
    assert meta['threshold'] == calibration['selected']['threshold']
    assert meta['candidate_filter'] == POLICY and meta['boundary_refiner']['threshold'] == 0.8
    assert meta['min_value_length'] == PROMPTGUARD_MIN_VALUE_LENGTH
    assert meta['calibration_reference'] == 'models/span-detector-v9' and not meta['production_ready']
    assert 'gate_exception' in meta['selection']
    for name in ['model.safetensors', 'boundary.pt', 'tokenizer.json', 'config.json']:
        assert (V10 / name).exists()


def test_length_guard_is_validation_neutral_and_v9_bundle_untouched():
    report = read('reports/ml_span/min-value-length.json')
    assert report['min_value_length'] == PROMPTGUARD_MIN_VALUE_LENGTH and report['gold_preservation']['suppressed'] == 0
    for result in report['checkpoints'].values():
        assert result['validation_passed'] and result['removed_tp_any_set'] == 0
    assert 'min_value_length' not in read('models/span-detector-v9/span_bundle.json')


def test_v10_adoption_rests_on_predeclared_holdout_rule():
    decision = read('configs/span-holdout-v11-decision.json')
    report = read('reports/ml_span/holdout-v11-evaluation.json')
    assert decision['candidate']['threshold'] == read('models/span-detector-v10/span_bundle.json')['threshold']
    assert report['decision']['candidate_better'] is True
    gates = read('reports/ml_span/context-v10-warm-regression.json')
    assert sum(c['passed'] for c in gates['checks']) == 10 and not gates['regression_passed']


def test_v10_and_holdout_splits_are_disjoint():
    v10 = {name: load(ROOT / f'data/span-context-v10/{name}.jsonl') for name in ('train', 'validation', 'test')}
    assert_disjoint(v10)
    assert_disjoint({'train': v10['train'], 'holdout_v11': load(ROOT / 'data/span-holdout-v11/test.jsonl')})
