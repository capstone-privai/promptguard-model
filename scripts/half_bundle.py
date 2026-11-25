"""Store a finished bundle's weights in bfloat16 to halve its size, for publishing in models/ (Git LFS).

  PYTHONPATH=src python scripts/half_bundle.py --src BUNDLE --out OUT \
      [--check-span VALIDATION.jsonl | --check-verifier CANDIDATES.jsonl.gz] [--device cuda]

Only model.safetensors changes: its floating tensors are rounded to bfloat16. Every other file is copied as it is.
The loaders keep computing in float32: MLSpanDetector loads a span bundle in float32 unless span_bundle.json sets
`torch_dtype`, and the verifier loader always loads float32. So the published bundle differs from the trained one
only by that rounding. A span bundle that already sets a half `torch_dtype` is refused (it is half already).

The check scores validation with the source bundle and with the halved one and records both under `half_storage`
in the bundle json: exact-span counts at the bundle's cutoff for a span bundle, MASK decisions at the deployment
threshold for a verifier bundle (ambiguous validation candidates). Counts only; no text or values are written.
"""
import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

WEIGHTS = 'model.safetensors'
SPAN_JSON, VERIFIER_JSON = 'span_bundle.json', 'bundle.json'


def bundle_json(path):
    for name in (SPAN_JSON, VERIFIER_JSON):
        if (Path(path) / name).exists():
            return name
    raise FileNotFoundError(f'{path} has neither {SPAN_JSON} nor {VERIFIER_JSON}')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def halve(src, out):
    """Copy the bundle with its floating weights in bfloat16; return (json name, metadata dict)."""
    src, out = Path(src), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty')
    name = bundle_json(src)
    meta = json.loads((src / name).read_text())
    if name == SPAN_JSON and meta.get('torch_dtype', 'float32') != 'float32':
        raise ValueError(f'{src} already loads in {meta["torch_dtype"]}')
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name != WEIGHTS:
            shutil.copy2(f, out / f.name)
    tensors = load_file(str(src / WEIGHTS))
    halved = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous() for k, v in tensors.items()}
    save_file(halved, str(out / WEIGHTS), metadata={'format': 'pt'})
    meta['half_storage'] = dict(
        dtype='bfloat16', computed_in='float32 (the loaders upcast)', source_weights_sha256=sha256(src / WEIGHTS),
        note='model.safetensors floating tensors rounded to bfloat16 to halve the size; everything else as trained')
    (out / name).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    return name, meta


def check_span(src, out, rows_path, device):
    from privai_model.ml_span.evaluation import measures
    from privai_model.ml_span.runtime import MLSpanDetector
    from privai_model.ml_span.schema import load
    rows = load(rows_path)
    gold = {(r['id'], s['start'], s['end']) for r in rows for s in r['spans']}
    result = {}
    for label, path in (('source', src), ('halved', out)):
        detector = MLSpanDetector(Path(path), device=device)
        detector.delimiter_policy = None  # as trained
        pred = {(r['id'], s['start'], s['end']) for r in rows for s in detector.scan_text(r['text'])}
        result[label] = {k: v for k, v in measures(gold, pred).items() if k in ('tp', 'fp', 'fn', 'precision', 'recall', 'f2')}
        result.setdefault('threshold', detector.threshold)
    return dict(kind='span', validation=str(rows_path), rows=len(rows), **result)


def check_verifier(src, out, candidates_path, device):
    from privai_model.inference.predict import load_model, predict_candidates
    with gzip.open(candidates_path, 'rt', encoding='utf-8') as f:
        records = [r for r in map(json.loads, f) if r['source']['split'] == 'validation']
    scores = {}
    for label, path in (('source', src), ('halved', out)):
        bundle = load_model(Path(path), device=device)
        scores[label] = {p['candidate_id']: p['p_mask'] for p in predict_candidates(bundle, records, batch_size=32)}
        threshold = bundle.threshold
    keys = scores['source'].keys()
    mask = {label: {k for k in keys if scores[label][k] >= threshold} for label in scores}
    truth = {r['candidate_id'] for r in records if r['label']['action'] == 'MASK'}
    counts = lambda m: dict(masked=len(m), true_masks=len(m & truth), missed_masks=len(truth - m))
    return dict(kind='verifier', candidates=str(candidates_path), validation_candidates=len(records), threshold=threshold,
                source=counts(mask['source']), halved=counts(mask['halved']),
                decisions_changed=len(mask['source'] ^ mask['halved']),
                max_abs_p_mask_change=max((abs(scores['source'][k] - scores['halved'][k]) for k in keys), default=0.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--out', required=True)
    check = p.add_mutually_exclusive_group()
    check.add_argument('--check-span', help='validation rows (jsonl) of the span model')
    check.add_argument('--check-verifier', help='candidates.jsonl.gz of the verifier (its validation split is used)')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA not available: stop (no CPU fallback)')
    name, meta = halve(args.src, args.out)
    if args.check_span:
        meta['half_storage']['check'] = check_span(args.src, args.out, args.check_span, args.device)
    elif args.check_verifier:
        meta['half_storage']['check'] = check_verifier(args.src, args.out, args.check_verifier, args.device)
    (Path(args.out) / name).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(meta['half_storage'], ensure_ascii=False))


if __name__ == '__main__':
    main()
