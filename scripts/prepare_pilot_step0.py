"""Team step-0 pilot: span rows for model A (hand + privesc) or model B (hand + LLM + privesc).

The team builds both training sets with promptguard-data metric-detection/scripts/build_ml_handoff.sh
(branch ml-train-pilot) and checks every file against the committed sha256. A and B differ only in
their train and val files; every other setting is shared so a score gap can be put on the LLM
templates. Train and val are kept whole, exactly as the team made them.

The pipeline needs a test split. The team scores A and B on its own test sets first, so neither is
put here: the test split is our hold-out v11 (a development set), and its scores are not used.
Values follow real credential formats and privesc holds SSH keys, so the output goes under runs/.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_context_v11 import team_rows, composition
from privai_model.io import write_json, write_jsonl
from privai_model.ml_span.schema import load, group_keys, digest
from privai_model.ml_span.splitter import assert_disjoint

SHARED_KINDS = ('value',)  # also set as shared_key_kinds in the pilot training config
FILES =[f'{kind}_{split}.jsonl' for kind in ('sessions', 'gold', 'labels') for split in ('train', 'val')]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--team', required=True, help='promptguard-data/metric-detection after build_ml_handoff.sh')
    p.add_argument('--model', required=True, choices=['A', 'B'])
    args = p.parse_args()
    team = Path(args.team)
    handoff = team / 'ml_handoff' / f'model_{args.model}'
    expected = json.loads((team / 'templates/ml_train/handoff_step0.json').read_text())['models'][args.model]['sha256']
    for name in FILES:
        if sha256(handoff / name) != expected[f'model_{args.model}/{name}']:
            raise ValueError(f'model_{args.model}/{name} does not match handoff_step0.json')
    out = ROOT / f'runs/data/span-pilot-step0-{args.model.lower()}'
    if out.exists():
        raise FileExistsError(out)
    rows = {split: team_rows(handoff / f'sessions_{split}.jsonl', handoff / f'gold_{split}.jsonl',
                             handoff / f'labels_{split}.jsonl', 'team_step0') for split in ('train', 'val')}
    splits = dict(train=rows['train'], validation=rows['val'], test=load(ROOT / 'data/span-holdout-v11/test.jsonl'))
    # The team keeps groups, texts and templates apart and lets common values recur across splits on purpose.
    # Check everything else and count the values instead of dropping rows, so A and B train on the team's files.
    assert_disjoint(splits, SHARED_KINDS)
    keys = {name: {k for r in split for k in group_keys(r) if k[0] in SHARED_KINDS} for name, split in splits.items()}
    shared = {f'{a}/{b}': len(keys[a] & keys[b]) for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))}
    out.mkdir(parents=True)
    for name, split in splits.items():
        write_jsonl(out / f'{name}.jsonl', split)
    sessions = {split: sum(1 for _ in open(handoff / f'sessions_{split}.jsonl')) for split in ('train', 'val')}
    write_json(out / 'manifest.json', dict(
        model=args.model, team_sessions=sessions, team_sha256={k: expected[f'model_{args.model}/{k}'] for k in FILES},
        composition={name: composition(split) for name, split in splits.items()},
        shared_keys=shared, fingerprint=digest(json.dumps(splits, sort_keys=True)),
        notice=('Team step-0 handoff, one row per prompt, instructions and tool_output item. Test is hold-out v11 '
                '(development, unused). Contains realistic fake credentials and SSH keys; do not commit.')))
    print(json.dumps(dict(model=args.model, sessions=sessions, composition={n: composition(s) for n, s in splits.items()},
                          shared_keys=shared), ensure_ascii=False))


if __name__ == '__main__':
    main()
