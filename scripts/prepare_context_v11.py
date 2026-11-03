"""Context v11: add the team's hand_v1 ML training data (promptguard-data ml_handoff) to the v10 union.

On the team's detection scorer the v10 recipe trailed v9 on the hand_v1 test split, mostly in
recall. The team builds train/val splits of the same corpus for the ML owner. v11 trains on
the v10 union plus hand_v1 train and selects on the v9 validation plus hand_v1 val. The
hand_v1 test split stays out of every v11 file; it was already used for diagnosis, so it is
reported as development only.

The team's values follow real credential formats, so the handoff README forbids committing
them. The output goes under runs/ (git-ignored). Rebuild it from a promptguard-data checkout:
ml_train_build.py and ml_train_export.py are deterministic, so every rebuild is byte-identical.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from privai_model.io import write_json, write_jsonl
from privai_model.ml_span.schema import load, group_keys, validate, digest
from privai_model.ml_span.splitter import assert_disjoint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/data/span-context-v11'
SOURCE = {'prompt': 'prompt', 'instructions': 'prompt', 'tool_output': 'tool_output'}  # tool_input is never scanned


def read(path):
    return [json.loads(line) for line in open(path, encoding='utf-8')]


def team_rows(sessions_path, gold_path, labels_path, prefix):
    """One row per scanned item: gold spans are secrets; gold and labelled non-gold intervals are registered values."""
    gold, labels = {}, {}
    for g in read(gold_path):
        gold.setdefault((g['session_id'], g['item_id']), []).append((g['span']['start'], g['span']['end']))
    for l in read(labels_path) if labels_path else []:
        labels.setdefault((l['session_id'], l['item_id']), []).append((l['start'], l['end']))
    rows = []
    for session in read(sessions_path):
        group = session['meta']['group_id']
        for item in session['items']:
            if item['channel'] not in SOURCE:
                continue
            key = (session['session_id'], item['item_id'])
            secrets = sorted(gold.get(key, []))
            values = set(secrets)
            for a, b in labels.get(key, []):
                if not any(max(a, c) < min(b, d) for c, d in values):
                    values.add((a, b))
            span = lambda a, b: dict(start=a, end=b, secret_type='SECRET')
            rows.append(validate(dict(
                schema_version='secret-span-v1', synthetic=True, id=f'{prefix}:{session["session_id"]}:{item["item_id"]}',
                text=item['text'], spans=[span(a, b) for a, b in secrets], value_spans=[span(a, b) for a, b in sorted(values)],
                source=SOURCE[item['channel']], language='mixed', family_id=f'{prefix}_{group}', template_id=f'{prefix}_{group}',
                origin=f'{prefix}:{group}', value_family=f'{prefix}_{group}',
                kind='positive' if secrets else 'hard_negative')))
    return rows


def drop_shared_values(rows, other):
    """Drop rows that reuse a value of another split. The team reuses common strings (weak passwords, user
    names, masks) across groups on purpose, but our train and validation splits must stay value-disjoint."""
    held = {k for r in other for k in group_keys(r) if k[0] == 'value'}
    kept = [r for r in rows if not any(k in held for k in group_keys(r) if k[0] == 'value')]
    return kept, len(rows) - len(kept)


def composition(rows):
    return dict(records=len(rows), positive=sum(bool(r['spans']) for r in rows), negative=sum(not r['spans'] for r in rows),
                gold_spans=sum(len(r['spans']) for r in rows), families=len({r['family_id'] for r in rows}),
                sources=dict(Counter(r['source'] for r in rows)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--team', required=True, help='promptguard-data/metric-detection with ml_handoff/ and the exported hand_v1 test')
    team = Path(p.parse_args().team)
    if OUT.exists():
        raise FileExistsError(OUT)
    handoff = team / 'ml_handoff'
    team_train = team_rows(handoff / 'sessions_train.jsonl', handoff / 'gold_train.jsonl', handoff / 'labels_train.jsonl', 'team_hand_v1')
    team_val = team_rows(handoff / 'sessions_val.jsonl', handoff / 'gold_val.jsonl', handoff / 'labels_val.jsonl', 'team_hand_v1')
    team_test = team_rows(team / 'data_test/sessions_mleval_hand_v1_test.jsonl',
                          team / 'data_answer/gold_mleval_hand_v1_test.jsonl', None, 'team_hand_v1')
    # Training keeps every team train row (the common-value cases are what v11 must learn); validation gives way.
    team_val, dropped_val = drop_shared_values(team_val, team_train)
    train = load(ROOT / 'data/span-context-v10/train.jsonl') + team_train
    validation = load(ROOT / 'data/span-context-v10/validation.jsonl') + team_val
    test = load(ROOT / 'data/span-holdout-v11/test.jsonl')
    assert_disjoint({'train': train, 'validation': validation, 'test': test})
    # The team test split is in no v11 file. Hold it to the team's own policy: groups, texts and templates are
    # disjoint; common values may recur by design and are counted, not removed.
    test_keys = {k for r in team_test for k in group_keys(r)}
    if any(k in test_keys for r in train + validation for k in group_keys(r) if k[0] != 'value'):
        raise ValueError('team test shares a group, text or template with v11 train or validation')
    shared_test_values = len({k for r in train + validation for k in group_keys(r) if k[0] == 'value'} &
                             {k for k in test_keys if k[0] == 'value'})
    # Team rows must not reuse a value, text or template from any earlier repository evaluation file.
    earlier = {k for path in sorted((ROOT / 'data').glob('span-*/*.jsonl')) if path.stem != 'train'
               for r in load(path) for k in group_keys(r) if k[0] != 'value_family'}
    if any(k in earlier for r in team_train + team_val for k in group_keys(r) if k[0] != 'value_family'):
        raise ValueError('team rows share a key with an earlier evaluation set')
    OUT.mkdir(parents=True)
    splits = dict(train=train, validation=validation, test=test)
    for name, rows in splits.items():
        write_jsonl(OUT / f'{name}.jsonl', rows)
    team_manifest = json.loads((handoff / 'manifest.json').read_text())
    write_json(OUT / 'manifest.json', dict(
        team_handoff=dict(files=team_manifest.get('files'), version=team_manifest.get('version')),
        team_rows=dict(train=composition(team_train), validation=composition(team_val), held_out_test=composition(team_test),
                       validation_rows_dropped_for_values_in_train=dropped_val,
                       common_values_shared_with_team_test=shared_test_values),
        composition={name: composition(rows) for name, rows in splits.items()},
        fingerprint=digest(json.dumps(splits, sort_keys=True)),
        notice=('Train: v10 union plus team hand_v1 train items (prompt, instructions, tool_output). Validation: v9 validation '
                'plus team hand_v1 val. Test: hold-out v11 (development). The team hand_v1 test split is in no v11 file. '
                'Contains realistic fake credentials; do not commit.')))
    print(json.dumps({name: composition(rows) for name, rows in splits.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
