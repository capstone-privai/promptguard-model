"""The server verifier: a Qwen decoder trained with LoRA as the candidate-v1 classifier (privai_model.verifier.lora).

Configs keep the mmBERT verifiers' data and selection and the Qwen span models' training; the markers are trained
rows that start apart; a run writes a bundle that inference and final evaluation load with the same scores; a run
cut off mid-epoch resumes to the same result. Tiny offline Qwen2 models and synthetic candidates only."""
import gzip
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from privai_model.data.adapter import load_examples
from privai_model.data.input_builder import MARKERS
from privai_model.data.split import restore_split
from privai_model.io import write_json
from privai_model.verifier.candidates import split_manifest

peft = pytest.importorskip('peft')
from privai_model.verifier import lora  # noqa: E402  (needs peft)

ROOT = Path(__file__).resolve().parents[2]
SIZES = ('1.5b', '7b', '14b')
PRECISION = {'torch_dtype', 'bf16', 'fp16'}


def config_file(name):
    return yaml.safe_load((ROOT / f'configs/{name}.yaml').read_text())


@pytest.mark.parametrize('size', SIZES)
def test_qwen_verifier_keeps_the_mmbert_verifiers_data_and_the_span_models_training(size):
    qwen = config_file(f'verifier-creddata-qwen{size}-bidir')
    mmbert = config_file('verifier-creddata-mmbert-small')
    span = config_file(f'token-span-creddata-qwen{size}-bidir')
    for key in ('data_path', 'split_manifest', 'seed', 'window_size', 'batch_size', 'class_weights', 'sampling',
                'character_features', 'recall_targets', 'deployment_recall_target', 'evaluate_test'):
        assert qwen[key] == mmbert[key], key
    for key in ('model_name', 'epochs', 'learning_rate', 'weight_decay', 'schedule', 'warmup_ratio', 'max_grad_norm',
                'lora', 'bidirectional', 'attn_implementation', 'gradient_checkpointing', *PRECISION):
        assert qwen.get(key) == span.get(key), key
    assert qwen['evaluate_test'] is False and qwen['output_dir'] == f'runs/verifier/creddata-qwen{size}-bidir'


def test_qwen_verifier_sizes_differ_only_in_the_model_and_precision():
    a = config_file('verifier-creddata-qwen1.5b-bidir')
    for size in SIZES[1:]:
        b = config_file(f'verifier-creddata-qwen{size}-bidir')
        differ = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        assert differ == {'model_name', 'model_version', 'output_dir'} | PRECISION


WORDS = ['<pad>', '<unk>', '[', ']', '[/', 'V', 'V_TRUNCATED', 'password', 'build', '=', 'id', 'Zx81Qv', 'Kp42Lm',
         'Tq55Rw', 'b7731', 'b2210', 'b9045', 'run', 'ok', 'TYPE', 'PASSWORD', 'LINE', 'BEFORE', 'AFTER']


def tiny_base(tmp_path, spare=8):
    """A tiny Qwen2 classifier and fast tokenizer saved like a Hub model. `spare` rows past the vocabulary hold the
    markers as Qwen's do; with spare=0 the embedding grows instead."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForSequenceClassification
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>', unk_token='<unk>')
    torch.manual_seed(0)
    model = Qwen2ForSequenceClassification(Qwen2Config(
        vocab_size=len(WORDS) + spare, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=128, num_labels=2, attn_implementation='sdpa'))
    path = tmp_path / f'tiny-qwen-{spare}'
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def record(i, split):
    secret = i % 2 == 1
    value = ['Zx81Qv', 'Kp42Lm', 'Tq55Rw'][i % 3] if secret else ['b7731', 'b2210', 'b9045'][i % 3]
    line = f'password = {value}' if secret else f'build id = {value}'
    return {'schema_version': 'candidate-v1', 'candidate_id': f'r{i}:0:{len(value)}', 'group_id': f'g{i}',
            'granularity': 'value', 'value': value,
            'line': {'text': line, 'value_start': line.index(value), 'value_end': len(line)},
            'context': {'before': ['run'], 'after': ['ok']}, 'type': 'PASSWORD',
            'source': {'dataset': 'synthetic-test-only', 'channel': 'tool_output', 'file_path': None, 'split': split},
            'detector': {'name': 'synthetic', 'rule_id': 'secret-assignment', 'raw_type': 'PASSWORD', 'score': None,
                         'strength': 'ambiguous'},
            'label': {'action': 'MASK' if secret else 'KEEP', 'provisional': False, 'reason': 'synthetic'}}


def candidate_data(tmp_path):
    records = [record(i, 'train' if i < 16 else 'validation' if i < 24 else 'test') for i in range(32)]
    data = tmp_path / 'data'
    data.mkdir()
    with gzip.open(data / 'candidates.jsonl.gz', 'wt', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    write_json(data / 'split_manifest.json', split_manifest(records))
    return data


def run_config(base, data, out, **extra):
    config = dict(model_name=str(base), revision=None, local_files_only=True, model_version='verifier-test-v0',
                  data_path=str(data / 'candidates.jsonl.gz'), split_manifest=str(data / 'split_manifest.json'),
                  output_dir=str(out), seed=1, split_ratios=[0.8, 0.1, 0.1], window_size=48, batch_size=4, epochs=2,
                  learning_rate=1e-2, weight_decay=0.0, schedule='cosine', warmup_ratio=0.1, max_grad_norm=1.0,
                  device='cpu', class_weights=None, sampling='none', character_features=False,
                  recall_targets=[0.95, 0.99], deployment_recall_target=0.99, evaluate_test=False,
                  bidirectional=True, attn_implementation='sdpa', log_every=1000, eval_batch_size=8,
                  lora=dict(r=4, alpha=8, dropout=0.1, target_modules=['q_proj', 'v_proj'], modules_to_save=['score']))
    config.update(extra)
    return config


@pytest.mark.parametrize('spare', [8, 0])
def test_marker_rows_start_apart_and_are_the_only_trained_embedding_rows(tmp_path, spare):
    base = tiny_base(tmp_path, spare)
    config = run_config(base, tmp_path, tmp_path / 'out')
    tokenizer, pieces = lora.load_tokenizer(config['model_name'], config)
    assert pieces['[V]'] == [WORDS.index('['), WORDS.index('V'), WORDS.index(']')]
    model = lora.load_classifier(config, tokenizer, pieces)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all('lora_' in n or '.score.' in n or 'trainable_tokens_delta' in n for n in trainable)
    assert {p.dtype for p in model.parameters() if p.requires_grad} == {torch.float32}
    delta = next(p for n, p in model.named_parameters() if 'trainable_tokens_delta' in n)
    assert delta.shape == (len(MARKERS), 32)
    rows = delta.detach()
    assert not torch.allclose(rows[0], rows[1]) and not torch.allclose(rows[0], rows[2])
    embedding = [p for n, p in model.named_parameters() if 'embed_tokens' in n and 'trainable_tokens' not in n]
    assert len(embedding) == 1 and not embedding[0].requires_grad
    assert torch.allclose(rows[0], embedding[0][pieces['[V]']].float().mean(0), atol=1e-6)


def validation_scores(run):
    return {row['candidate_id']: row['p_mask'] for row in map(json.loads, (run / 'validation_predictions.jsonl').open())}


@pytest.mark.parametrize('precision', [{}, {'torch_dtype': 'bfloat16', 'bf16': True}])
def test_a_run_writes_a_bundle_that_inference_and_final_evaluation_load_with_the_same_scores(tmp_path, precision):
    from privai_model.inference.predict import load_model, predict_examples
    data, out = candidate_data(tmp_path), tmp_path / 'run'
    report = lora.train(run_config(tiny_base(tmp_path), data, out, **precision), log=lambda _: None)
    assert report['test'] is None and set(report['thresholds']) == {'0.95', '0.99'}
    bundle_json = json.loads((out / 'model' / 'bundle.json').read_text())
    assert bundle_json['adapter'] == 'lora' and bundle_json['best_epoch'] in (1, 2)
    assert bundle_json['threshold_status'] == 'validation_selected'
    assert len(json.loads((out / 'history.json').read_text())) == 2
    assert not (out / 'test_predictions.jsonl').exists()

    bundle = load_model(out / 'model', device='cpu')
    assert bundle.threshold == bundle_json['thresholds']['0.99']
    assert bundle.amp == (torch.bfloat16 if precision else None)
    splits = restore_split(load_examples(data / 'candidates.jsonl.gz'), json.loads((out / 'split_manifest.json').read_text()))
    again = {row['candidate_id']: row['p_mask'] for row in predict_examples(bundle, splits['validation'], 3)}
    saved = validation_scores(out)
    assert again.keys() == saved.keys()
    assert all(abs(again[k] - saved[k]) < 1e-3 for k in saved)

    spec = importlib.util.spec_from_file_location('final_evaluate', ROOT / 'scripts/final_evaluate.py')
    final = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(final)
    result = final.finalize(data / 'candidates.jsonl.gz', out, out / 'final', device='cpu')
    assert result['thresholds'] == bundle_json['thresholds']
    assert {v['records'] for v in result['test'].values()} == {len(splits['test'])}


class Interrupted(Exception):
    pass


def test_a_run_cut_off_mid_epoch_resumes_to_the_uninterrupted_result(tmp_path):
    data, base = candidate_data(tmp_path), tiny_base(tmp_path)
    settings = dict(log_every=1, save_every_minutes=1e-9)  # a resumable checkpoint after every batch
    lora.train(run_config(base, data, tmp_path / 'whole', **settings), log=lambda _: None)

    def cut_at_step_3(line):
        if '"step": 3,' in line and not cut_at_step_3.done:
            cut_at_step_3.done = True
            raise Interrupted
    cut_at_step_3.done = False
    config = run_config(base, data, tmp_path / 'cut', **settings)
    with pytest.raises(Interrupted):
        lora.train(config, log=cut_at_step_3)
    assert json.loads((tmp_path / 'cut' / 'last' / 'state.json').read_text())['step'] == 2
    with pytest.raises(FileExistsError):
        lora.train(config, log=lambda _: None)  # without resume a used output_dir is refused
    lines = []
    lora.train(config, resume=True, log=lines.append)
    assert json.loads(next(line for line in lines if '"resumed"' in line))['batch'] == 2

    whole, cut = (json.loads((tmp_path / name / 'history.json').read_text()) for name in ('whole', 'cut'))
    for a, b in zip(whole, cut, strict=True):
        assert a['epoch'] == b['epoch']
        assert a['train_loss'] == pytest.approx(b['train_loss'], abs=1e-6)
        assert a['validation_loss'] == pytest.approx(b['validation_loss'], abs=1e-6)
    a, b = validation_scores(tmp_path / 'whole'), validation_scores(tmp_path / 'cut')
    assert all(abs(a[k] - b[k]) < 1e-6 for k in a)
    with pytest.raises(FileExistsError, match='finished'):
        lora.train(config, resume=True, log=lambda _: None)


def test_settings_the_lora_verifier_cannot_take_are_refused(tmp_path):
    config = run_config(tmp_path, tmp_path, tmp_path / 'out')
    for change, message in ((dict(lora=None), 'lora'), (dict(sampling='balanced'), 'sampling'),
                            (dict(character_features=True), 'character_features'),
                            (dict(deployment_recall_target=0.9), 'recall')):
        with pytest.raises(ValueError, match=message):
            lora.validate(dict(config, **change))
