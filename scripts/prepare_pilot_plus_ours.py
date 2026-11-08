"""Pilot follow-up: model B's train plus our hard negatives (B+HN), to test the team data's false-positive gap.

Models trained on the team step-0 data fire on secret-like strings our sets mark as not secret (hashes,
request IDs, redaction marks, values a context calls examples). This adds every secret-free hard-negative
row of data/span-context-v10/train.jsonl to B's train. Validation (the team's B val) and test stay as in
runs/data/span-pilot-step0-b, so epoch and cutoff are chosen exactly as for B.
"""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_context_v11 import composition
from privai_model.io import write_json, write_jsonl
from privai_model.ml_span.schema import load, digest
from privai_model.ml_span.splitter import assert_disjoint

SHARED_KINDS = ('value',)
SOURCE = 'data/span-context-v10/train.jsonl'


def main():
    base = ROOT / 'runs/data/span-pilot-step0-b'
    out = ROOT / 'runs/data/span-pilot-step0-bhn'
    if out.exists():
        raise FileExistsError(out)
    negatives = [r for r in load(ROOT / SOURCE) if r['kind'] == 'hard_negative' and not r['spans']]
    splits = {name: load(base / f'{name}.jsonl') for name in ('train', 'validation', 'test')}
    splits['train'] = splits['train'] + negatives
    assert_disjoint(splits, SHARED_KINDS)
    out.mkdir(parents=True)
    for name, split in splits.items():
        write_jsonl(out / f'{name}.jsonl', split)
    manifest = json.loads((base / 'manifest.json').read_text())
    write_json(out / 'manifest.json', dict(
        model='B+HN', base=dict(data=str(base.relative_to(ROOT)), fingerprint=manifest['fingerprint']),
        added=dict(source=SOURCE, rows=len(negatives), filter="kind == 'hard_negative' and no spans"),
        team_sessions=manifest['team_sessions'], composition={n: composition(s) for n, s in splits.items()},
        fingerprint=digest(json.dumps(splits, sort_keys=True)),
        notice='B train plus our secret-free hard negatives; B val and test unchanged. Contains team values; do not commit.'))
    print(json.dumps(dict(added=len(negatives), composition={n: composition(s) for n, s in splits.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
