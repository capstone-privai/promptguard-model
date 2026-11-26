"""CredData-trajectory v1 plus NoseyParker-trajectory v1 train: span rows for every model retrained with NP added.

  PYTHONPATH=src python scripts/prepare_creddata_np_traj.py --np-src NoseyParker-trajectory/ \
      [--creddata runs/data/span-creddata-traj-v1] [--out runs/data/span-creddata-np-traj-v1]

CredData-trajectory keeps the split every earlier result used: runs/data/span-creddata-traj-v1 (built by
scripts/prepare_creddata_traj.py, fingerprint 19c26b1e...) is copied unchanged, so the CredData part of validation is
the same 1,200 gold spans the earlier models were selected on.

NoseyParker-trajectory v1 (from the team, not in Git) is Claude Code sessions run on Nosey Parker rule examples whose
secrets were replaced by format-keeping generated values. The zip holds only its train part. Its test (21 rule files)
stays with the team as the hold-out, so NP adds nothing to test here. Its README asks for splits by meta.group_id
(the Nosey Parker rule file); the ko/en pair of a task shares a source file and so a group.

NP train is split by group: groups are ordered by a seeded hash; a group with more than 10% of the NP gold stays in
train; validation takes groups until it holds 20% of the NP gold; the rest is train. One row per prompt, instructions
and tool_output item, the same as CredData-trajectory (tool_input is not scanned). Nothing is dropped.

Some rule files hold the same snippet around different values: NP groups whose rows with spans share a text or a
canonical template (the text with every value replaced) form one unit, so no row with a span shares its context with
another split. Units, not single groups, are what the rule above assigns.

Writes train / validation / test.jsonl (the models read these), validation_creddata.jsonl and validation_np.jsonl
(the two parts of validation, for scoring each source on its own) and manifest.json. Values look like real
credentials: the output goes under runs/ and is never committed.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_context_v11 import team_rows, composition
from privai_model.io import write_json, write_jsonl
from privai_model.ml_span.schema import load, digest, group_keys

FILES = ('sessions_train.jsonl', 'gold_train.jsonl', 'labels_train.jsonl', 'manifest.json')
PREFIX = 'noseyparker_traj'
CREDDATA_FINGERPRINT = '19c26b1e60c435f5b16afe702ea860cc5aa712811cf0564a6c1712c9d94fb40c'
SHARE = 0.20        # of the NP gold, for validation
LARGE_GROUP = 0.10  # an NP group with more of the NP gold than this stays in train
SPLITS = ('train', 'validation', 'test')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_of(row):
    return row['family_id'][len(PREFIX) + 1:]


def linked_groups(rows):
    """{group: unit}: NP groups whose rows with spans share a text or canonical template (the text with every value
    replaced) are one unit. Different rule files can hold the same snippet around different values; splitting them
    apart would put the same context, with other values, in train and validation."""
    parent = {group_of(r): group_of(r) for r in rows}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    owner = {}
    for r in rows:
        if not r['spans']:
            continue
        for key in group_keys(r):
            if key[0] in ('text', 'canonical_template'):
                a, b = find(owner.setdefault(key, group_of(r))), find(group_of(r))
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return {g: find(g) for g in parent}


def assign(gold_by_group, seed):
    total = sum(gold_by_group.values())
    order = sorted(gold_by_group, key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    split, held = {}, 0
    for g in order:
        n = gold_by_group[g]
        if n <= LARGE_GROUP * total and held < SHARE * total:
            split[g] = 'validation'
            held += n
        else:
            split[g] = 'train'
    return split


def overlap(splits, key):
    keys = {name: {k for r in rows for k in key(r)} for name, rows in splits.items()}
    return {f'{a}/{b}': len(keys[a] & keys[b]) for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--np-src', required=True, help='NoseyParker-trajectory folder')
    p.add_argument('--creddata', default=str(ROOT / 'runs/data/span-creddata-traj-v1'))
    p.add_argument('--out', default=str(ROOT / 'runs/data/span-creddata-np-traj-v1'))
    p.add_argument('--seed', default='noseyparker-traj-v1')
    args = p.parse_args()
    src, cred_dir, out = Path(args.np_src), Path(args.creddata), Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    cred = {name: load(cred_dir / f'{name}.jsonl') for name in SPLITS}
    if digest(json.dumps(cred, sort_keys=True)) != CREDDATA_FINGERPRINT:
        raise ValueError(f'{cred_dir} is not the CredData-trajectory split the earlier models used')
    rows = team_rows(src / 'sessions_train.jsonl', src / 'gold_train.jsonl', src / 'labels_train.jsonl', PREFIX)
    unit_of = linked_groups(rows)
    gold_by_unit = Counter({u: 0 for u in unit_of.values()})
    for r in rows:
        gold_by_unit[unit_of[group_of(r)]] += len(r['spans'])
    unit_split = assign(gold_by_unit, args.seed)
    split_of = {g: unit_split[u] for g, u in unit_of.items()}
    nps = {name: [r for r in rows if split_of[group_of(r)] == name] for name in SPLITS}
    splits = {name: cred[name] + nps[name] for name in SPLITS}
    ids = [r['id'] for split in splits.values() for r in split]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate row IDs across the two sources')
    positive_text = lambda r: [digest(r['text'])] if r['spans'] else []
    values = lambda r: [digest(r['text'][s['start']:s['end']]) for s in r['spans']]
    out.mkdir(parents=True)
    for name, split in splits.items():
        write_jsonl(out / f'{name}.jsonl', split)
    write_jsonl(out / 'validation_creddata.jsonl', cred['validation'])
    write_jsonl(out / 'validation_np.jsonl', nps['validation'])
    manifest = dict(
        creddata=dict(data=str(cred_dir), fingerprint=CREDDATA_FINGERPRINT, note='copied unchanged'),
        noseyparker=dict(source={name: sha256(src / name) for name in FILES}, seed=args.seed,
                         rule=f'by meta.group_id (Nosey Parker rule file), groups whose rows with spans share a text or '
                              f'canonical template joined into one unit; units over {LARGE_GROUP:.0%} of the NP gold stay '
                              f'in train; validation takes units in seeded hash order until it holds {SHARE:.0%} of the '
                              f'NP gold; NP test stays with the team',
                         groups={name: sum(1 for g in split_of.values() if g == name) for name in ('train', 'validation')},
                         linked_units=sorted(Counter(unit_of.values()).values(), reverse=True)[:10],
                         composition={name: composition(nps[name]) for name in ('train', 'validation') if nps[name]}),
        composition={name: composition(split) for name, split in splits.items()},
        # Reported, not removed: boilerplate (system prompt blocks, tool messages) recurs in every split.
        shared_positive_texts=overlap(splits, positive_text), shared_secret_values=overlap(splits, values),
        np_vs_creddata_shared_secret_values=len({k for r in rows for k in values(r)}
                                                & {k for s in cred.values() for r in s for k in values(r)}),
        fingerprint=digest(json.dumps(splits, sort_keys=True)),
        notice=('CredData-trajectory v1 (split unchanged) plus NoseyParker-trajectory v1 train split by rule file. '
                'Test is CredData-trajectory test only and is not scored for these models; the NP test is the team\'s '
                'hold-out. Contains realistic fake credentials and possibly unlabelled real ones; do not commit.'))
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('composition', 'shared_positive_texts', 'shared_secret_values',
                                                'np_vs_creddata_shared_secret_values', 'fingerprint')}
                     | dict(np=manifest['noseyparker']), ensure_ascii=False))


if __name__ == '__main__':
    main()
