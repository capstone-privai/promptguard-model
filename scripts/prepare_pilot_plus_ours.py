"""Pilot follow-ups: model B's data plus rows from our data.

  prepare_pilot_plus_ours.py bhn    B train + every secret-free hard negative of data/span-context-v10/train.jsonl
  prepare_pilot_plus_ours.py bv10   B train + all of v10 train (positives and hard negatives);
                                    validation = B val + v10 validation
  prepare_pilot_plus_ours.py bcred  B train + all of CredData-trajectory v1 train (runs/data/span-creddata-traj-v1,
                                    built by scripts/colab_prepare_creddata_traj.sh); validation = B val + its validation
  prepare_pilot_plus_ours.py bcrednp  as bcred with NoseyParker-trajectory v1 added (runs/data/span-creddata-np-traj-v1,
                                      built by scripts/colab_prepare_creddata_np_traj.sh)

Models trained on the team step-0 data fire on secret-like strings our sets mark as not secret (hashes,
request IDs, redaction marks, values a context calls examples). B+HN adds only our hard negatives, and the
model learned to stop detecting anything written in our formats: negatives without positives from the same
templates teach the format, not the context. B+v10 adds the hard negatives together with the positives they
were generated beside. Its validation holds both sources so epoch and cutoff are chosen for both. B+CredData does
the same with CredData-trajectory (real repositories' credentials in agent trajectories) in place of v10; its test
split stays closed here. Test stays B's placeholder (our hold-out v11, see prepare_pilot_step0.py).
"""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_context_v11 import composition
from privai_model.io import write_json, write_jsonl
from privai_model.ml_span.schema import load, digest, group_keys
from privai_model.ml_span.splitter import assert_disjoint

SHARED_KINDS = ('value',)
V10 = 'data/span-context-v10'
VARIANTS = {
    'bhn': dict(model='B+HN', source=V10, train=lambda r: r['kind'] == 'hard_negative' and not r['spans'],
                validation=False, filter="kind == 'hard_negative' and no spans",
                notice='B train plus our secret-free hard negatives; B val and test unchanged.'),
    'bv10': dict(model='B+v10', source=V10, train=lambda r: True, validation=True, filter='all rows',
                 notice='B train plus all of our v10 train; validation is B val plus v10 validation; B test unchanged.'),
    'bcred': dict(model='B+CredData', source='runs/data/span-creddata-traj-v1', train=lambda r: True, validation=True,
                  filter='all rows', leakage='span rows',
                  notice='B train plus all of CredData-trajectory v1 train; validation is B val plus '
                  'CredData-trajectory validation; B test unchanged, CredData-trajectory test not added.'),
    'bcrednp': dict(model='B+CredData+NP', source='runs/data/span-creddata-np-traj-v1', train=lambda r: True,
                    validation=True, filter='all rows', leakage='span rows',
                    notice='B train plus all of CredData-trajectory v1 train and the NoseyParker-trajectory v1 train '
                    'rule files; validation is B val plus CredData-trajectory validation plus the NP validation rule '
                    'files; B test unchanged, CredData-trajectory test not added, the NP test is the team\'s hold-out.'),
}


def check_span_rows(splits):
    """No row with a span shares a group key (other than SHARED_KINDS) with a row in another split.

    Rows without spans may: CredData-trajectory was split by session, so some secret-free tool outputs recur in its
    train and validation. Its split is kept as every CredData-trajectory result used it, and no row is dropped.
    Returns how many keys rows without spans share across splits."""
    where = {}
    for name, rows in splits.items():
        for row in rows:
            for key in group_keys(row):
                if key[0] not in SHARED_KINDS:
                    where.setdefault(key, set()).add(name)
    for rows in splits.values():
        for row in rows:
            if row['spans'] and any(len(where[k]) > 1 for k in group_keys(row) if k[0] not in SHARED_KINDS):
                raise ValueError('split leakage detected in a row with spans')
    return sum(len(names) > 1 for names in where.values())


def main(variant):
    spec = VARIANTS[variant]
    base = ROOT / 'runs/data/span-pilot-step0-b'
    out = ROOT / f'runs/data/span-pilot-step0-{variant}'
    if out.exists():
        raise FileExistsError(out)
    source = ROOT / spec['source']
    splits = {name: load(base / f'{name}.jsonl') for name in ('train', 'validation', 'test')}
    added = dict(train=[r for r in load(source / 'train.jsonl') if spec['train'](r)])
    if spec['validation']:
        added['validation'] = load(source / 'validation.jsonl')
    for name, rows in added.items():
        splits[name] = splits[name] + rows
    leakage = dict(policy=spec.get('leakage', 'all rows'))
    if leakage['policy'] == 'span rows':
        leakage['span_free_keys_shared_across_splits'] = check_span_rows(splits)
    else:
        assert_disjoint(splits, SHARED_KINDS)
    out.mkdir(parents=True)
    for name, split in splits.items():
        write_jsonl(out / f'{name}.jsonl', split)
    manifest = json.loads((base / 'manifest.json').read_text())
    write_json(out / 'manifest.json', dict(
        model=spec['model'], base=dict(data=str(base.relative_to(ROOT)), fingerprint=manifest['fingerprint']),
        added=dict(source=spec['source'], rows={n: len(r) for n, r in added.items()}, filter=spec['filter']),
        leakage_check=leakage,
        team_sessions=manifest['team_sessions'], composition={n: composition(s) for n, s in splits.items()},
        fingerprint=digest(json.dumps(splits, sort_keys=True)), notice=spec['notice'] + ' Contains team values; do not commit.'))
    print(json.dumps(dict(added={n: len(r) for n, r in added.items()}, composition={n: composition(s) for n, s in splits.items()}),
                     ensure_ascii=False))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'bhn')
