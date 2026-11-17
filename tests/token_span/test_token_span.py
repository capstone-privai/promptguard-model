"""The token-classifier comparison: shared settings, cutoff selection, resumable batch order, group split,
bidirectional attention for the decoder."""
import importlib.util
import random
from pathlib import Path

import pytest
import torch
import yaml

from privai_model.ml_span.evaluation import measures
from privai_model.token_span.train import bidirectional, epoch_batches, select_cutoff

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


def test_bidirectional_qwen_differs_from_causal_qwen_only_in_attention():
    a, b = config('qwen'), config('qwen-bidir')
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {'output_dir', 'bidirectional', 'attn_implementation'}
    assert b['bidirectional'] is True and b['attn_implementation'] == 'sdpa'


def tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForTokenClassification
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64, num_labels=3, attn_implementation='sdpa')
    return Qwen2ForTokenClassification(cfg).eval()


def logits(model, ids, mask=None):
    ids = torch.tensor(ids)
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=torch.ones_like(ids) if mask is None else torch.tensor(mask)).logits


def test_bidirectional_lets_early_tokens_see_later_ones():
    before, after = [[5, 6, 7, 8]], [[5, 6, 7, 9]]
    causal = tiny_qwen()
    assert torch.allclose(logits(causal, before)[0, :3], logits(causal, after)[0, :3])
    both = bidirectional(tiny_qwen())
    assert not torch.allclose(logits(both, before)[0, 0], logits(both, after)[0, 0])


def test_bidirectional_ignores_padding_and_survives_lora():
    peft = pytest.importorskip('peft')
    model = peft.get_peft_model(bidirectional(tiny_qwen()), peft.LoraConfig(
        task_type='TOKEN_CLS', r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'], modules_to_save=['score']))
    model.eval()
    alone = logits(model, [[5, 6, 7]])[0]
    padded = logits(model, [[5, 6, 7, 0, 0], [1, 2, 3, 4, 5]], [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])[0, :3]
    assert torch.allclose(alone, padded, atol=1e-5)
    assert not torch.allclose(logits(model, [[5, 6, 7, 8]])[0, 0], logits(model, [[5, 6, 7, 9]])[0, 0])


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
