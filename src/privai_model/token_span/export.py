"""Turn a token-span training run (token_span.train) into an ml-span bundle that MLSpanDetector loads as it is.

A run leaves one set of weights per epoch (`epoch-N/`: a LoRA adapter with the head for Qwen, a whole model for
mmBERT) and `selection.json` with the epoch and cutoff chosen on validation. The bundle holds:

- the chosen epoch as one token classifier: the LoRA merged into its base, saved in `dtype` (default: the base's
  training dtype, so bfloat16 for 7B and 14B), with the tokenizer;
- `span_bundle.json` (ml-span-bundle-v1): max_length, stride, the cutoff as `threshold`, `best_epoch`,
  `bidirectional`, `torch_dtype`, the base model and the Hub commit its weights came from, and how it was trained.

The epoch and cutoff always come from selection.json; there is no way to pick others here.

Merging changes the numbers slightly: the LoRA delta is rounded into bfloat16 weights, and a 4-bit (QLoRA) run is
merged into a bfloat16 base, not the 4-bit one it was trained on. `check_validation` scores validation with the
exported bundle the way MLSpanDetector runs it and records the counts next to the training-time ones.
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

from privai_model.io import write_json
from privai_model.ml_span.bio import LABELS
from privai_model.ml_span.evaluation import measures
from privai_model.ml_span.schema import load
from privai_model.token_span.train import DTYPES, gold_keys, load_model

BUNDLE_FILE = 'span_bundle.json'


def read_run(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    selection = json.loads((run_dir / 'selection.json').read_text())
    run = json.loads((run_dir / 'run.json').read_text()) if (run_dir / 'run.json').exists() else {}
    return config, selection, run


def chosen_entry(selection):
    epoch = selection['selected']['epoch']
    entry = next(h for h in selection['history'] if h['epoch'] == epoch)
    if entry['selected']['threshold'] != selection['selected']['threshold']:
        raise ValueError('selection.json disagrees with its own history')
    return entry


def export(run_dir, out_dir, dtype=None, overrides=None, log=print):
    """Write the bundle to out_dir (which must not exist or be empty) and return its span_bundle.json."""
    run_dir, out = Path(run_dir), Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty')
    config, selection, run = read_run(run_dir)
    config = dict(config, **(overrides or {}))
    entry = chosen_entry(selection)
    epoch, threshold = selection['selected']['epoch'], selection['selected']['threshold']
    weights = run_dir / f'epoch-{epoch}'
    trained_dtype = config.get('torch_dtype', 'float32')
    dtype = dtype or trained_dtype
    if dtype not in DTYPES:
        raise ValueError(f'dtype must be one of {sorted(DTYPES)}')
    # The commit the base weights came from; the config may still say `revision: null` for runs before pinning.
    revision = run.get('revision_resolved') or config.get('revision')
    lora = bool(config.get('lora'))
    loading = dict(config, revision=revision, torch_dtype=dtype, load_in_4bit=False, bf16=dtype == 'bfloat16',
                   fp16=False, gradient_checkpointing=False, bidirectional=False)  # the mask is a runtime hook
    model = load_model(loading, adapter=str(weights))
    if lora:
        model = model.merge_and_unload()
    model = model.to(DTYPES[dtype]).eval()
    model.config.id2label = dict(LABELS)
    model.config.label2id = {v: k for k, v in LABELS.items()}
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    for extra in ('adapter_config.json', 'adapter_model.safetensors'):  # a merged bundle carries no adapter
        if (out / extra).exists():
            raise RuntimeError(f'{extra} was saved; the LoRA was not merged')
    picked = entry['selected']
    meta = {
        'schema_version': 'ml-span-bundle-v1',
        'max_length': config['max_length'],
        'stride': config['stride'],
        'threshold': threshold,
        'threshold_source': 'token-span selection.json: exact-span F2 on validation at every observed span score, '
                            'precision tie-break, earlier epoch on ties',
        'best_epoch': epoch,
        'bidirectional': bool(config.get('bidirectional')),
        'torch_dtype': dtype,
        'backbone': config['model_name'],
        'revision': revision,
        'label_mapping': {str(k): v for k, v in LABELS.items()},
        'weights': 'lora-merged' if lora else 'full-fine-tune',
        'training': {
            'torch_dtype': trained_dtype,
            'load_in_4bit': bool(config.get('load_in_4bit')),
            'autocast': run.get('amp'),
            'lora': config.get('lora'),
            'seed': config.get('seed'),
            'epochs_run': len(selection['history']),
            'gpu': run.get('gpu'),
            'torch': run.get('torch'),
        },
        'validation': {'rows': selection.get('validation', {}).get('rows'),
                       **{k: picked.get(k) for k in ('tp', 'fp', 'fn', 'precision', 'recall', 'f2')}},
        'source_run': run_dir.name,
        'production_ready': False,
    }
    write_json(out / BUNDLE_FILE, meta)
    log(json.dumps(dict(event='exported', epoch=epoch, threshold=threshold, dtype=dtype,
                        bidirectional=meta['bidirectional'], weights=meta['weights'])))
    return meta


def check_validation(bundle_dir, data_dir, device='cpu', dtype=None, log=print):
    """Score validation with the bundle as MLSpanDetector runs it; record counts beside the training-time ones."""
    from privai_model.ml_span.runtime import MLSpanDetector
    bundle_dir = Path(bundle_dir)
    meta = json.loads((bundle_dir / BUNDLE_FILE).read_text())
    rows = load(Path(data_dir) / 'validation.jsonl')
    detector = MLSpanDetector(bundle_dir, device, dtype=dtype)
    kept = {(row['id'], s['start'], s['end']) for row in rows for s in detector.scan_text(row['text'])}
    exact = measures(gold_keys(rows), kept)
    trained = {k: meta['validation'][k] for k in ('tp', 'fp', 'fn')}
    check = dict(split='validation', rows=len(rows), device=str(detector.device),
                 dtype=str(next(detector.model.parameters()).dtype).replace('torch.', ''), exact=exact,
                 matches_training={k: exact[k] == trained[k] for k in trained})
    meta['export_check'] = check
    write_json(bundle_dir / BUNDLE_FILE, meta)
    log(json.dumps(dict(event='checked', training=trained, bundle={k: exact[k] for k in trained})))
    return check
