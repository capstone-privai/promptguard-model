"""SFT conversion and pre-training checks. Reports hold counts and IDs, never text."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from privai_model.ml_span.schema import load, group_keys
from . import format as fmt

TRAIN_DIR = 'data/span-context-v9'

# Evaluation sets used with v9. `validation` selects the operating point; everything
# else is frozen comparison only. Historical *_development sets were inspected during
# v4-v9 development, so they are development checks, not independent test results.
EVAL_SETS = {
    'validation': 'data/span-context-v9/validation.jsonl',
    'fresh_test': 'data/span-context-v9/test.jsonl',
    'v3_development': 'data/span-refinement-v3/test.jsonl',
    'v1_development': 'data/span-refinement-v1/test.jsonl',
    'original_development': 'data/span-synthetic-v1/test.jsonl',
    'negative_challenge': 'data/span-negative-challenge-v1/test.jsonl',
    'v4_development': 'data/span-context-v4/test.jsonl',
    'v6_development': 'data/span-context-v6/test.jsonl',
    'v7_development': 'data/span-context-v7/test.jsonl',
}


def sft_record(row):
    return dict(id=row['id'], messages=fmt.messages(row['text'], fmt.target(row)))


def roundtrip_report(rows):
    gold = recovered = 0
    failures = []
    for row in rows:
        expected = {(s['start'], s['end']) for s in row['spans']}
        got = fmt.check_roundtrip(row)
        gold += len(expected)
        recovered += len(expected & got)
        if got != expected:
            failures.append(row['id'])
    return dict(rows=len(rows), gold_spans=gold, recovered_exactly=recovered, failed_ids=failures)


def leakage_report(train, evaluations):
    """Count shared grouping/text/template/value keys between train and each eval set."""
    train_keys = {}
    for row in train:
        for key in group_keys(row):
            train_keys.setdefault(key, row['id'])
    report = {}
    for name, rows in evaluations.items():
        by_kind = Counter()
        rows_hit = set()
        for row in rows:
            for key in group_keys(row):
                if key in train_keys:
                    by_kind[key[0]] += 1
                    rows_hit.add(row['id'])
        report[name] = dict(rows=len(rows), rows_sharing_any_key=len(rows_hit), shared_keys_by_kind=dict(by_kind))
    return report


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, output_dir, train_dir=TRAIN_DIR, eval_sets=None):
    root = Path(root)
    output_dir = Path(output_dir)
    eval_sets = dict(EVAL_SETS if eval_sets is None else eval_sets)
    train_path = root / train_dir / 'train.jsonl'
    train = load(train_path)
    evaluations = {name: load(root / path) for name, path in eval_sets.items()}
    roundtrip = {'train': roundtrip_report(train)}
    roundtrip.update({name: roundtrip_report(rows) for name, rows in evaluations.items()})
    if any(r['failed_ids'] for r in roundtrip.values()):
        raise ValueError('gold span roundtrip failed; refusing to write SFT data')
    leakage = leakage_report(train, evaluations)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'train.jsonl').open('w', encoding='utf-8') as f:
        for row in train:
            f.write(json.dumps(sft_record(row), ensure_ascii=False) + '\n')
    targets = Counter(len(r['spans']) for r in train)
    manifest = dict(
        schema_version='qwen-span-sft-v1', answer_format=fmt.SCHEMA_VERSION,
        system_prompt_sha256=hashlib.sha256(fmt.SYSTEM_PROMPT.encode()).hexdigest(),
        source_metadata_in_prompt=False,
        note=('Only text is given to the model, like the v9 ML detector. The `source` field is '
              'excluded because some training rows use source="negative", which would leak the label.'),
        train=dict(path=str(train_dir) + '/train.jsonl', sha256=sha256_file(train_path), rows=len(train),
                   positives=sum(bool(r['spans']) for r in train), spans_per_row=dict(sorted(targets.items()))),
        evaluation_sets={name: dict(path=path, sha256=sha256_file(root / path), rows=len(evaluations[name]))
                         for name, path in eval_sets.items()},
        roundtrip=roundtrip, leakage=leakage)
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return manifest
