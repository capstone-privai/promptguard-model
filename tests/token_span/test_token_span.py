"""The token-classifier comparison: shared settings, cutoff selection, resumable batch order, group split."""
import importlib.util
import random
from pathlib import Path

import yaml

from privai_model.ml_span.evaluation import measures
from privai_model.token_span.train import epoch_batches, select_cutoff

ROOT = Path(__file__).resolve().parents[2]
MODEL_SPECIFIC = {'model_name', 'revision', 'output_dir', 'epochs', 'patience', 'learning_rate', 'weight_decay',
                  'schedule', 'warmup_ratio', 'gradient_checkpointing', 'lora', 'save_every', 'save_every_minutes'}


def config(name):
    return yaml.safe_load((ROOT / f'configs/token-span-creddata-{name}.yaml').read_text())


def test_mmbert_and_qwen_share_data_windows_and_selection():
    a, b = config('mmbert'), config('qwen')
    differ = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differ <= MODEL_SPECIFIC
    for key in ('data_dir', 'seed', 'batch_size', 'class_weights', 'max_length', 'stride', 'fp16', 'eval_batch_size'):
        assert a[key] == b[key]


def row(rid, spans):
    return dict(id=rid, spans=[dict(start=s, end=e) for s, e in spans])


def test_select_cutoff_matches_brute_force():
    rng = random.Random(0)
    rows = [row(f'r{i}', [(j * 10, j * 10 + 5) for j in range(rng.randint(0, 3))]) for i in range(40)]
    predictions = {}
    for r in rows:
        spans = [dict(start=s['start'], end=s['end'] + rng.choice([0, 0, 1]), score=round(rng.random(), 2)) for s in r['spans']]
        spans += [dict(start=50, end=55, score=round(rng.random(), 2))] * rng.randint(0, 1)
        predictions[r['id']] = spans
    raw, best, _ = select_cutoff(rows, predictions)
    gold = {(r['id'], s['start'], s['end']) for r in rows for s in r['spans']}
    brute = []
    for t in sorted({s['score'] for v in predictions.values() for s in v}):
        m = measures(gold, {(rid, s['start'], s['end']) for rid, v in predictions.items() for s in v if s['score'] >= t})
        brute.append(((m['f2'] or 0), (m['precision'] or 0), t))
    assert (best['f2'], best['precision'], best['threshold']) == max(brute)
    assert raw['tp'] + raw['fn'] == len(gold)


def test_epoch_batches_replay_the_same_order_within_an_epoch():
    lengths = [random.Random(1).randint(1, 512) for _ in range(103)]
    first = epoch_batches(lengths, 4, 110, 0)
    assert first == epoch_batches(lengths, 4, 110, 0)
    assert first != epoch_batches(lengths, 4, 110, 1)
    assert sorted(i for b in first for i in b) == list(range(103))


def test_group_split_keeps_large_groups_in_train_and_fills_test_and_validation():
    spec = importlib.util.spec_from_file_location('prep', ROOT / 'scripts/prepare_creddata_traj.py')
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    gold = {f'g{i}': n for i, n in enumerate([400] + [20] * 100)}
    split = prep.assign(gold, 'seed')
    assert split['g0'] == 'train'
    total = sum(gold.values())
    for name in ('test', 'validation'):
        held = sum(n for g, n in gold.items() if split[g] == name)
        assert prep.SHARE * total <= held < prep.SHARE * total + 20
