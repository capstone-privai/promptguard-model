"""CredData-trajectory v1: span rows split 80/10/10 by component for the token-classifier comparison.

The data (from the team, not in Git) is Claude Code sessions run on CredData train files whose secrets were
replaced by generated values; gold is where those values appear. Its README asks for splits by meta.group_id
(component): augmented copies and the ko/en pair of a task share their original's group, so they stay together.

One row per prompt, instructions and tool_output item, as PromptGuard scans them (tool_input is not scanned and
the team scorer does not score it). Non-gold candidates from labels_train.jsonl are registered as values, not
secrets, the same as the team's step-0 rows.

Split: groups are ordered by a seeded hash. A group holding more than 3% of the gold stays in train, so one large
component cannot dominate validation or test. Test, then validation, take groups until each holds 10% of the gold;
the rest is train. Nothing is dropped. Test is a hold-out: it is scored only after both models are frozen.

Values look like real credentials and source files may hold unlabelled real ones, so the output goes under runs/.
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
from privai_model.ml_span.schema import digest

FILES = ('sessions_train.jsonl', 'gold_train.jsonl', 'labels_train.jsonl', 'manifest.json')
PREFIX = 'creddata_traj'
SHARE = 0.10        # of the gold, for test and for validation each
LARGE_GROUP = 0.03  # a group with more of the gold than this stays in train


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_of(row):
    return row['family_id'][len(PREFIX) + 1:]


def assign(gold_by_group, seed):
    total = sum(gold_by_group.values())
    order = sorted(gold_by_group, key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    split, held = {}, Counter()
    for g in order:
        n = gold_by_group[g]
        if n > LARGE_GROUP * total:
            split[g] = 'train'
        elif held['test'] < SHARE * total:
            split[g] = 'test'
        elif held['validation'] < SHARE * total:
            split[g] = 'validation'
        else:
            split[g] = 'train'
        held[split[g]] += n
    return split


def overlap(splits, key):
    keys = {name: {k for r in rows for k in key(r)} for name, rows in splits.items()}
    return {f'{a}/{b}': len(keys[a] & keys[b]) for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='CredData-trajectory folder')
    p.add_argument('--out', default=str(ROOT / 'runs/data/span-creddata-traj-v1'))
    p.add_argument('--seed', default='creddata-traj-v1')
    args = p.parse_args()
    src, out = Path(args.src), Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    rows = team_rows(src / 'sessions_train.jsonl', src / 'gold_train.jsonl', src / 'labels_train.jsonl', PREFIX)
    gold_by_group = Counter()
    for r in rows:
        gold_by_group[group_of(r)] += len(r['spans'])
    split_of = assign(gold_by_group, args.seed)
    splits = {name: [r for r in rows if split_of[group_of(r)] == name] for name in ('train', 'validation', 'test')}
    positive_text = lambda r: [digest(r['text'])] if r['spans'] else []
    values = lambda r: [digest(r['text'][s['start']:s['end']]) for s in r['spans']]
    out.mkdir(parents=True)
    for name, split in splits.items():
        write_jsonl(out / f'{name}.jsonl', split)
    manifest = dict(
        source={name: sha256(src / name) for name in FILES}, seed=args.seed,
        rule=f'by meta.group_id; groups over {LARGE_GROUP:.0%} of the gold stay in train; test then validation take '
             f'groups in seeded hash order until each holds {SHARE:.0%} of the gold',
        groups={name: sum(1 for g in split_of.values() if g == name) for name in splits},
        composition={name: composition(split) for name, split in splits.items()},
        # Reported, not removed: boilerplate (system prompt blocks, tool messages) recurs in every split.
        shared_positive_texts=overlap(splits, positive_text), shared_secret_values=overlap(splits, values),
        fingerprint=digest(json.dumps(splits, sort_keys=True)),
        notice=('CredData-trajectory v1, one row per prompt, instructions and tool_output item. Test is a hold-out, '
                'scored only after both models are frozen. Contains realistic fake credentials and possibly '
                'unlabelled real ones from CredData sources; do not commit.'))
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('groups', 'composition', 'shared_positive_texts', 'shared_secret_values',
                                                'fingerprint')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
