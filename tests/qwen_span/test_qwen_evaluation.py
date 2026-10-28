import json
from pathlib import Path
import pytest
from privai_model.ml_span.schema import load
from privai_model.qwen_span import data as qdata
from privai_model.qwen_span import evaluation as qeval

ROOT = Path(__file__).resolve().parents[2]


def rows():
    base = dict(schema_version='secret-span-v1', synthetic=True, source='log', language='en',
                template_id='t', origin='o', value_family='v', kind='positive')
    return [dict(base, id='a', family_id='f1', text='k=abc', spans=[dict(start=2, end=5, secret_type='SECRET')]),
            dict(base, id='b', family_id='f2', text='k=xyz', spans=[], kind='hard_negative')]


def pred(spans, error=None):
    return dict(spans=[dict(start=a, end=b, score=s) for a, b, s in spans], error=error, counts={}, latency_ms=1.0)


def test_score_counts_negatives_and_invalid_outputs():
    p = {'a': pred([(2, 5, .9)]), 'b': pred([(2, 5, .4)])}
    result = qeval.score(rows(), p, 0.0)
    assert (result['exact']['tp'], result['exact']['fp'], result['exact']['fn']) == (1, 1, 0)
    assert result['negative_rows_with_findings'] == 1
    assert qeval.score(rows(), p, 0.5)['exact']['fp'] == 0
    p['a'] = pred([], 'invalid_json')
    assert qeval.score(rows(), p)['invalid_outputs'] == {'invalid_json': 1}


def test_threshold_selection_uses_observed_scores():
    p = {'a': pred([(2, 5, .9)]), 'b': pred([(2, 5, .4)])}
    chosen = qeval.select_threshold(rows(), p)
    assert chosen['selected']['threshold'] == .9 and chosen['selected']['fp'] == 0
    assert qeval.select_threshold(rows(), p, minimum_precision=1.0)['eligible']


def test_incomplete_predictions_rejected():
    with pytest.raises(ValueError):
        qeval.score(rows(), {'a': pred([])})


def test_compare_with_v9_partitions_true_positives():
    p = {'a': pred([(2, 5, 1.)]), 'b': pred([(0, 1, 1.)])}
    c = qeval.compare_with_v9(rows(), p, 0.0, v9=set())
    assert c['qwen_only_tp'] == 1 and c['qwen_new_fp'] == 1 and c['v9_only_tp'] == 0
    c = qeval.compare_with_v9(rows(), p, 0.0, v9={('a', 2, 5), ('b', 3, 5)})
    assert c['both_tp'] == 1 and c['v9_fp_avoided_by_qwen'] == 1


def test_v9_rebuild_matches_saved_report_on_every_set():
    report = json.loads((ROOT / 'reports/ml_span/context-v9-comparison.json').read_text())['datasets']
    for name, path in qdata.EVAL_SETS.items():
        data = load(ROOT / path)
        predicted = qeval.v9_predictions(data, report[name])
        stored = report[name]['candidate']['ml_only']
        assert qeval.measures(qeval.gold_keys(data), predicted)['tp'] == stored['tp']


def test_prepare_roundtrips_all_gold_and_finds_no_leakage(tmp_path):
    manifest = qdata.prepare(ROOT, tmp_path)
    for result in manifest['roundtrip'].values():
        assert result['recovered_exactly'] == result['gold_spans'] and not result['failed_ids']
    assert all(r['rows_sharing_any_key'] == 0 for r in manifest['leakage'].values())
    first = json.loads((tmp_path / 'train.jsonl').read_text().splitlines()[0])
    assert 'source' not in first['messages'][1]['content']
