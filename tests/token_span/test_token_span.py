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


# The size series (mmBERT-base; Qwen 1.5B, 7B, 14B) and the bfloat16 / 4-bit loading the larger Qwen models need.

PRECISION = {'torch_dtype', 'bf16', 'fp16'}


@pytest.mark.parametrize('name, precision', [('qwen1.5b-bidir', False), ('qwen7b-bidir', True), ('qwen14b-bidir', True)])
def test_qwen_size_series_differs_from_the_05b_run_only_in_size_and_precision(name, precision):
    a, b = config('qwen-bidir'), config(name)
    differ = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differ == {'model_name', 'revision', 'output_dir'} | (PRECISION if precision else set())
    assert b['model_name'].startswith('Qwen/Qwen2.5-Coder-') and b['bidirectional'] is True
    if precision:
        assert b['torch_dtype'] == 'bfloat16' and b['bf16'] is True and not b.get('fp16')


def test_mmbert_base_differs_from_mmbert_small_only_in_the_model():
    a, b = config('mmbert'), config('mmbert-base')
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {'model_name', 'revision', 'output_dir'}
    assert b['model_name'] == 'jhu-clsp/mmBERT-base'


def test_base_loading_rejects_unsupported_combinations():
    from privai_model.token_span.train import base_loading
    assert base_loading({}) == dict(torch_dtype=torch.float32)
    assert base_loading({'torch_dtype': 'bfloat16', 'bf16': True}) == dict(torch_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='lora'):
        base_loading({'load_in_4bit': True})
    with pytest.raises(ValueError, match='bf16'):
        base_loading({'torch_dtype': 'bfloat16', 'fp16': True})


def test_amp_dtype_keeps_float16_on_cuda_only():
    from privai_model.token_span.train import amp_dtype
    cpu = torch.device('cpu')
    assert amp_dtype({'fp16': True}, cpu) is None
    assert amp_dtype({'bf16': True}, cpu) == torch.bfloat16
    assert amp_dtype({}, cpu) is None


def tiny_checkpoint(tmp_path):
    """A tiny Qwen2 token classifier and a fast tokenizer saved like a Hub model, so load_model and train run offline."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    words = ['<pad>', '<unk>', 'password', '=', 'token', 'user', 'name', 'value', 'run', 'ok', 'Zx81Qv', 'Kp42Lm']
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>', unk_token='<unk>')
    path = tmp_path / 'tiny-qwen'
    model = tiny_qwen()
    model.config.vocab_size = len(words)
    model.resize_token_embeddings(len(words))
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def span_rows(n, prefix):
    rows = []
    for i in range(n):
        secret = 'Zx81Qv' if i % 2 else 'Kp42Lm'
        text = f'user name = value run ok password = {secret}' if i % 3 else 'user name = value run ok'
        spans = [dict(start=text.index(secret), end=len(text), secret_type='SECRET')] if secret in text else []
        rows.append(dict(schema_version='secret-span-v1', synthetic=True, id=f'{prefix}{i}', text=text, spans=spans,
                         value_spans=spans, source='tool_output', language='en', family_id=f'f{i}', template_id='t',
                         origin='o', value_family='v', kind='positive' if spans else 'hard_negative'))
    return rows


def lora_config(path, **extra):
    return dict(model_name=str(path), revision=None, local_files_only=True, bidirectional=True,
                attn_implementation='sdpa', lora=dict(r=4, alpha=8, dropout=0.0, target_modules=['q_proj', 'v_proj'],
                                                      modules_to_save=['score']), **extra)


def test_bfloat16_base_keeps_trainable_weights_float32_and_stays_bidirectional(tmp_path):
    pytest.importorskip('peft')
    from privai_model.token_span.train import load_model
    model = load_model(lora_config(tiny_checkpoint(tmp_path), torch_dtype='bfloat16', bf16=True))
    trainable = {p.dtype for p in model.parameters() if p.requires_grad}
    frozen = {p.dtype for p in model.parameters() if not p.requires_grad}
    assert trainable == {torch.float32} and frozen == {torch.bfloat16}
    model.eval()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        first = logits(model, [[2, 3, 10]])[0, 0]
        changed = logits(model, [[2, 3, 11]])[0, 0]
    assert not torch.allclose(first.float(), changed.float())


def test_train_runs_end_to_end_with_a_bfloat16_base(tmp_path):
    pytest.importorskip('peft')
    import json
    from privai_model.io import write_jsonl
    from privai_model.token_span.train import train
    data = tmp_path / 'data'
    data.mkdir()
    write_jsonl(data / 'train.jsonl', span_rows(12, 't'))
    write_jsonl(data / 'validation.jsonl', span_rows(6, 'v'))
    config = lora_config(tiny_checkpoint(tmp_path), torch_dtype='bfloat16', bf16=True, data_dir=str(data),
                         output_dir=str(tmp_path / 'out'), device='cpu', seed=1, epochs=1, patience=1, batch_size=4,
                         learning_rate=1e-3, class_weights=[1.0, 2.0, 2.0], max_length=16, stride=4,
                         eval_batch_size=8, log_every=1000)
    selection = train(config, log=lambda _: None)
    assert selection['selected']['epoch'] == 1
    run = json.loads((tmp_path / 'out' / 'run.json').read_text())
    assert run['amp'] == 'bfloat16' and 'revision_resolved' in run


def test_bidirectional_also_covers_qwen3_dense_models():
    """Qwen3 (dense, full attention in every layer) takes the same padding-only mask; Qwen3.5 / Qwen3-Next do not
    (3 of 4 layers are linear attention, which the mask cannot make bidirectional)."""
    from transformers import Qwen3Config, Qwen3ForTokenClassification
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=40, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=64, num_labels=3,
                      attn_implementation='sdpa')
    causal = Qwen3ForTokenClassification(cfg).eval()
    assert torch.allclose(logits(causal, [[5, 6, 7, 8]])[0, 0], logits(causal, [[5, 6, 7, 9]])[0, 0])
    both = bidirectional(Qwen3ForTokenClassification(cfg).eval())
    assert not torch.allclose(logits(both, [[5, 6, 7, 8]])[0, 0], logits(both, [[5, 6, 7, 9]])[0, 0])


def trained_run(tmp_path, **extra):
    """A one-epoch bidirectional LoRA run on tiny data, as scripts/token_span.py train leaves it."""
    from privai_model.io import write_jsonl
    from privai_model.token_span.train import train
    data = tmp_path / 'data'
    data.mkdir()
    write_jsonl(data / 'train.jsonl', span_rows(12, 't'))
    write_jsonl(data / 'validation.jsonl', span_rows(6, 'v'))
    config = lora_config(tiny_checkpoint(tmp_path), data_dir=str(data), output_dir=str(tmp_path / 'run'), device='cpu',
                         seed=1, epochs=1, patience=1, batch_size=4, learning_rate=1e-2, class_weights=[1.0, 2.0, 2.0],
                         max_length=16, stride=4, eval_batch_size=8, log_every=1000, **extra)
    train(config, log=lambda _: None)
    return config


def test_exported_bundle_scores_like_the_run_and_needs_the_bidirectional_flag(tmp_path):
    pytest.importorskip('peft')
    import json
    from privai_model.ml_span.runtime import MLSpanDetector
    from privai_model.ml_span.schema import load
    from privai_model.token_span.export import export
    from privai_model.token_span.train import load_model, load_tokenizer, predict
    config = trained_run(tmp_path)
    selection = json.loads((tmp_path / 'run' / 'selection.json').read_text())
    meta = export(tmp_path / 'run', tmp_path / 'bundle', log=lambda _: None)
    assert (meta['best_epoch'], meta['threshold']) == (selection['selected']['epoch'], selection['selected']['threshold'])
    assert meta['bidirectional'] is True and meta['torch_dtype'] == 'float32' and meta['weights'] == 'lora-merged'
    assert not (tmp_path / 'bundle' / 'adapter_config.json').exists()
    rows = load(tmp_path / 'data' / 'validation.jsonl')
    model = load_model(dict(config, gradient_checkpointing=False), adapter=str(tmp_path / 'run' / 'epoch-1'))
    expected = predict(model, load_tokenizer(config), rows, 16, 4, torch.device('cpu'))

    def same(detector):
        for row in rows:
            got, want = detector.scan_text(row['text'], threshold=0.0), expected[row['id']]
            if [(s['start'], s['end']) for s in got] != [(s['start'], s['end']) for s in want]:
                return False
            if any(abs(a['score'] - b['score']) > 1e-4 for a, b in zip(got, want)):
                return False
        return True

    def early_sees_late(detector):
        return not torch.allclose(logits(detector.model, [[2, 3, 10, 9]])[0, 0], logits(detector.model, [[2, 3, 10, 8]])[0, 0])

    detector = MLSpanDetector(tmp_path / 'bundle')
    assert detector.threshold == meta['threshold'] and same(detector) and early_sees_late(detector)
    # Without the flag the same weights load with eager attention and run causally.
    meta.pop('bidirectional')
    (tmp_path / 'bundle' / 'span_bundle.json').write_text(json.dumps(meta))
    assert not early_sees_late(MLSpanDetector(tmp_path / 'bundle'))


def test_bfloat16_export_loads_in_bfloat16_unless_the_caller_asks_otherwise(tmp_path):
    pytest.importorskip('peft')
    from privai_model.ml_span.runtime import MLSpanDetector
    from privai_model.token_span.export import check_validation, export
    trained_run(tmp_path, torch_dtype='bfloat16', bf16=True)
    meta = export(tmp_path / 'run', tmp_path / 'bundle', log=lambda _: None)
    assert meta['torch_dtype'] == 'bfloat16' and meta['training']['autocast'] == 'bfloat16'
    detector = MLSpanDetector(tmp_path / 'bundle')
    assert {p.dtype for p in detector.model.parameters()} == {torch.bfloat16}
    assert all(isinstance(s['score'], float) for s in detector.scan_text('user name = value password = Zx81Qv', 0.0))
    wide = MLSpanDetector(tmp_path / 'bundle', dtype='float32')
    assert {p.dtype for p in wide.model.parameters()} == {torch.float32}
    with pytest.raises(RuntimeError, match='initialization failed'):
        MLSpanDetector(tmp_path / 'bundle', dtype='int8')
    check = check_validation(tmp_path / 'bundle', tmp_path / 'data', log=lambda _: None)
    assert check['rows'] == 6 and check['dtype'] == 'bfloat16' and set(check['matches_training']) == {'tp', 'fp', 'fn'}
    with pytest.raises(FileExistsError):
        export(tmp_path / 'run', tmp_path / 'bundle', log=lambda _: None)


def test_runtime_keeps_older_bundles_eager_and_float32():
    from privai_model.ml_span.runtime import MLSpanDetector
    bundle = ROOT / 'models' / 'span-pilot-step0-bv10-seed112'
    if not (bundle / 'model.safetensors').exists():
        pytest.skip('published pilot bundle weights unavailable')
    detector = MLSpanDetector(bundle)
    assert detector.model.config._attn_implementation == 'eager'
    assert {p.dtype for p in detector.model.parameters()} == {torch.float32}
    with pytest.raises(RuntimeError, match='initialization failed'):
        MLSpanDetector(tokenizer=detector.tokenizer, model=detector.model, dtype='bfloat16')
