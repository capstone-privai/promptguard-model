"""Trajectory-train-combined v1, the team's third training set: span rows for the models retrained on it.

  PYTHONPATH=src python scripts/prepare_traj_combined.py --src Trajectory-train-combined/ \
      [--earlier runs/data/span-creddata-np-traj-v1] [--out runs/data/span-traj-combined-v1]

The zip (from the team, not in Git) joins the train parts of three trajectory handoffs, all labelled with the
2026-11-27 v2 rules (fragments of 16 characters or more, encoded and decoded forms, parts derived from private keys):
CredData-trajectory v2 (207 groups, the same groups as v1), NoseyParker-trajectory v2 (62 rule files, the same as v1)
and infra-config-trajectory v1 train (594 groups, new). Its README asks for splits by meta.group_id: augmented CredData
copies and the ko/en pair of a task share their original's group.

Split:
- CredData and NoseyParker groups keep the split they had in runs/data/span-creddata-np-traj-v1 (fingerprint
  c7d0de5d..., built by scripts/prepare_creddata_np_traj.py): CredData train / validation / test, NoseyParker train /
  validation. Only the labels changed, so validation holds the same groups as before, and the CredData test groups stay
  a hold-out that no model trains on. The NoseyParker test stays with the team.
- infra-config is new and gets train and validation only; its test stays with the team. Groups whose rows with spans
  share a text or a canonical template (the text with every value replaced) form one unit, across all three sources. A
  unit that holds a CredData or NoseyParker group takes that group's split. The other units are ordered by a seeded
  hash; a unit with more than 3% of the infra-config gold stays in train, and validation takes units until it holds 10%
  of the infra-config gold.

One row per prompt, instructions and tool_output item, the same as every earlier trajectory split (tool_input is not
scanned). Nothing is dropped. Row IDs and groups use the earlier prefixes (creddata_traj, noseyparker_traj) plus
infra_config_traj, so the trainers' per-source counts name the three sources.

Writes train / validation / test.jsonl (the models read these), validation_<source>.jsonl for each source, and
manifest.json. Values look like real credentials: the output goes under runs/ and is never committed.
"""
import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_context_v11 import team_rows, composition
from privai_model.io import write_json, write_jsonl
from privai_model.ml_span.schema import load, digest, group_keys

FILES = ('sessions_train.jsonl', 'gold_train.jsonl', 'labels_train.jsonl', 'manifest.json', 'README.md')
EARLIER_FINGERPRINT = 'c7d0de5dc012c8040e17cfa2027956090c8dd90dda56d3caea4746e9ec0c8b8e'
SOURCES = {  # meta.source -> row ID prefix
    'creddata-derived-agent-pilot': 'creddata_traj',
    'noseyparker-derived-agent': 'noseyparker_traj',
    'infra-config-derived-agent': 'infra_config_traj',
}
NEW = 'infra_config_traj'
SHARE = 0.10        # of the infra-config gold, for validation
LARGE_GROUP = 0.03  # an infra-config unit with more of its gold than this stays in train
SPLITS = ('train', 'validation', 'test')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_by_source(src, tmp):
    """Cut the combined files by meta.source, line for line, so each source goes through team_rows (the row builder
    every earlier trajectory split used) on its own."""
    prefix_of, outs = {}, {}
    for prefix in SOURCES.values():
        (tmp / prefix).mkdir()
        outs[prefix] = {n: open(tmp / prefix / n, 'w', encoding='utf-8') for n in FILES[:3]}
    try:
        with open(src / 'sessions_train.jsonl', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    session = json.loads(line)
                    prefix = prefix_of[session['session_id']] = SOURCES[session['meta']['source']]
                    outs[prefix]['sessions_train.jsonl'].write(line if line.endswith('\n') else line + '\n')
        for name in FILES[1:3]:
            with open(src / name, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        outs[prefix_of[json.loads(line)['session_id']]][name].write(line if line.endswith('\n') else line + '\n')
    finally:
        for files in outs.values():
            for f in files.values():
                f.close()
    return {p: team_rows(tmp / p / FILES[0], tmp / p / FILES[1], tmp / p / FILES[2], p) for p in SOURCES.values()}


def linked_units(rows):
    """{family: unit}: families (groups) whose rows with spans share a text or canonical template are one unit, across
    sources. Splitting them apart would put the same context, with other values, in two splits."""
    parent = {r['family_id']: r['family_id'] for r in rows}

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
                a, b = find(owner.setdefault(key, r['family_id'])), find(r['family_id'])
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return {g: find(g) for g in parent}


def assign(gold_by_unit, seed):
    """Validation takes units in seeded hash order until it holds SHARE of the gold; large units stay in train."""
    total = sum(gold_by_unit.values())
    order = sorted(gold_by_unit, key=lambda u: hashlib.sha256(f'{seed}:{u}'.encode()).hexdigest())
    split, held = {}, 0
    for u in order:
        n = gold_by_unit[u]
        if n <= LARGE_GROUP * total and held < SHARE * total:
            split[u] = 'validation'
            held += n
        else:
            split[u] = 'train'
    return split


def split_families(rows, earlier_split, seed):
    """{family: split}. Families seen in the earlier split keep it; a unit holding any of them takes their split (the
    earlier split has no unit across splits; one would be an error); the other units are assigned by assign()."""
    unit_of = linked_units(rows)
    fixed = {}
    for family, unit in unit_of.items():
        if family in earlier_split:
            if fixed.setdefault(unit, earlier_split[family]) != earlier_split[family]:
                raise ValueError(f'unit {unit} links groups from two earlier splits')
        elif not family.startswith(NEW + '_'):
            raise ValueError(f'{family} is not in the earlier split')
    gold = Counter()
    for r in rows:
        if unit_of[r['family_id']] not in fixed:
            gold[unit_of[r['family_id']]] += len(r['spans'])
    free = assign(gold, seed)
    return {f: fixed.get(u) or free[u] for f, u in unit_of.items()}, unit_of, fixed


def overlap(splits, key):
    keys = {name: {k for r in rows for k in key(r)} for name, rows in splits.items()}
    return {f'{a}/{b}': len(keys[a] & keys[b]) for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='Trajectory-train-combined folder')
    p.add_argument('--earlier', default=str(ROOT / 'runs/data/span-creddata-np-traj-v1'))
    p.add_argument('--out', default=str(ROOT / 'runs/data/span-traj-combined-v1'))
    p.add_argument('--seed', default='trajectory-train-combined-v1')
    args = p.parse_args()
    src, earlier_dir, out = Path(args.src), Path(args.earlier), Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    earlier = {name: load(earlier_dir / f'{name}.jsonl') for name in SPLITS}
    if digest(json.dumps(earlier, sort_keys=True)) != EARLIER_FINGERPRINT:
        raise ValueError(f'{earlier_dir} is not the CredData + NoseyParker split the earlier models used')
    earlier_split = {}
    for name, split in earlier.items():
        for r in split:
            if earlier_split.setdefault(r['family_id'], name) != name:
                raise ValueError(f"{r['family_id']} is in two earlier splits")
    del earlier
    with tempfile.TemporaryDirectory() as tmp:
        by_source = split_by_source(src, Path(tmp))
    rows = [r for rs in by_source.values() for r in rs]
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate row IDs')
    split_of, unit_of, fixed = split_families(rows, earlier_split, args.seed)
    splits = {name: [r for r in rows if split_of[r['family_id']] == name] for name in SPLITS}
    positive_text = lambda r: [digest(r['text'])] if r['spans'] else []
    values = lambda r: [digest(r['text'][s['start']:s['end']]) for s in r['spans']]
    new_units = {unit_of[f] for f in split_of if f.startswith(NEW + '_')}
    out.mkdir(parents=True)
    for name, split in splits.items():
        write_jsonl(out / f'{name}.jsonl', split)
    for prefix in SOURCES.values():
        write_jsonl(out / f'validation_{prefix}.jsonl', [r for r in splits['validation'] if r['id'].startswith(prefix + ':')])
    manifest = dict(
        source=dict(files={name: sha256(src / name) for name in FILES},
                    version=json.loads((src / 'manifest.json').read_text())['version']),
        earlier=dict(data=str(earlier_dir), fingerprint=EARLIER_FINGERPRINT,
                     note='CredData and NoseyParker groups keep their split; only the labels are new'),
        rule=(f'infra-config by meta.group_id; groups whose rows with spans share a text or canonical template are one '
              f'unit across sources; a unit holding an earlier CredData or NoseyParker group takes its split; other units '
              f'over {LARGE_GROUP:.0%} of the infra-config gold stay in train; validation takes units in seeded hash '
              f'order until it holds {SHARE:.0%} of the infra-config gold; infra-config and NoseyParker tests stay with '
              f'the team'),
        seed=args.seed,
        infra_config=dict(units=len(new_units),
                          linked_to_earlier_groups=sorted({f for f in split_of if f.startswith(NEW + '_') and unit_of[f] in fixed}),
                          largest_units=sorted(Counter(unit_of[f] for f in split_of if f.startswith(NEW + '_')).values(),
                                               reverse=True)[:10]),
        composition={name: composition(split) for name, split in splits.items()},
        by_source={prefix: {name: composition([r for r in split if r['id'].startswith(prefix + ':')])
                            for name, split in splits.items()} for prefix in SOURCES.values()},
        # Reported, not removed: boilerplate (system prompt blocks, tool messages) recurs in every split.
        shared_positive_texts=overlap(splits, positive_text), shared_secret_values=overlap(splits, values),
        fingerprint=digest(json.dumps(splits, sort_keys=True)),
        notice=('Trajectory-train-combined v1 (CredData, NoseyParker, infra-config trajectories; v2 labels). CredData and '
                'NoseyParker keep the earlier split; infra-config is split by group into train and validation. Test is '
                'the earlier CredData test groups only and is not scored for these models. Contains realistic fake '
                'credentials and possibly unlabelled real ones; do not commit.'))
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('composition', 'shared_positive_texts', 'shared_secret_values',
                                                'infra_config', 'fingerprint')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
