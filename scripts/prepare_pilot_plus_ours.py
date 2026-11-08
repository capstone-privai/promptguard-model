"""Pilot follow-ups: model B's data plus rows from our v10 data.

  prepare_pilot_plus_ours.py bhn    B train + every secret-free hard negative of data/span-context-v10/train.jsonl
  prepare_pilot_plus_ours.py bv10   B train + all of v10 train (positives and hard negatives);
                                    validation = B val + v10 validation

Models trained on the team step-0 data fire on secret-like strings our sets mark as not secret (hashes,
request IDs, redaction marks, values a context calls examples). B+HN adds only our hard negatives, and the
model learned to stop detecting anything written in our formats: negatives without positives from the same
templates teach the format, not the context. B+v10 adds the hard negatives together with the positives they
were generated beside. Its validation holds both sources so epoch and cutoff are chosen for both. Test stays
B's placeholder (our hold-out v11, see prepare_pilot_step0.py).
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
SOURCE = 'data/span-context-v10'
VARIANTS = {
    'bhn': dict(model='B+HN', train=lambda r: r['kind'] == 'hard_negative' and not r['spans'], validation=False,
                filter="kind == 'hard_negative' and no spans",
                notice='B train plus our secret-free hard negatives; B val and test unchanged.'),
    'bv10': dict(model='B+v10', train=lambda r: True, validation=True, filter='all rows',
                 notice='B train plus all of our v10 train; validation is B val plus v10 validation; B test unchanged.'),
}


def main(variant):
    spec = VARIANTS[variant]
    base = ROOT / 'runs/data/span-pilot-step0-b'
    out = ROOT / f'runs/data/span-pilot-step0-{variant}'
    if out.exists():
        raise FileExistsError(out)
    splits = {name: load(base / f'{name}.jsonl') for name in ('train', 'validation', 'test')}
    added = dict(train=[r for r in load(ROOT / SOURCE / 'train.jsonl') if spec['train'](r)])
    if spec['validation']:
        added['validation'] = load(ROOT / SOURCE / 'validation.jsonl')
    for name, rows in added.items():
        splits[name] = splits[name] + rows
    assert_disjoint(splits, SHARED_KINDS)
    out.mkdir(parents=True)
    for name, split in splits.items():
        write_jsonl(out / f'{name}.jsonl', split)
    manifest = json.loads((base / 'manifest.json').read_text())
    write_json(out / 'manifest.json', dict(
        model=spec['model'], base=dict(data=str(base.relative_to(ROOT)), fingerprint=manifest['fingerprint']),
        added=dict(source=SOURCE, rows={n: len(r) for n, r in added.items()}, filter=spec['filter']),
        team_sessions=manifest['team_sessions'], composition={n: composition(s) for n, s in splits.items()},
        fingerprint=digest(json.dumps(splits, sort_keys=True)), notice=spec['notice'] + ' Contains team values; do not commit.'))
    print(json.dumps(dict(added={n: len(r) for n, r in added.items()}, composition={n: composition(s) for n, s in splits.items()}),
                     ensure_ascii=False))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'bhn')
