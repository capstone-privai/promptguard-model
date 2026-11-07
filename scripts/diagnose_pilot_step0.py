"""Break a pilot model's validation errors down by secret type, template source, channel and non-secret kind.

Validation only: the team scores its test sets first. Gold types and non-secret kinds come from the team's
val files (gold_val, labels_val); the report holds counts only, never text or values.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from privai_model.ml_span.runtime import MLSpanDetector
from privai_model.ml_span.schema import load


def origin(group):
    return 'hand' if group.startswith('hand') else 'privesc' if group.startswith('privesc') else 'llm'


def read(path):
    return [json.loads(line) for line in open(path, encoding='utf-8')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model', required=True, choices=['A', 'B'])
    p.add_argument('--team', required=True, help='promptguard-data/metric-detection after build_ml_handoff.sh')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threshold', type=float, help='defaults to the bundle cutoff')
    p.add_argument('--output')
    args = p.parse_args()
    handoff = Path(args.team) / 'ml_handoff' / f'model_{args.model}'
    rows = load(ROOT / f'runs/data/span-pilot-step0-{args.model.lower()}/validation.jsonl')
    gold_type, kinds, groups = {}, defaultdict(list), {}
    for g in read(handoff / 'gold_val.jsonl'):
        gold_type[(f"team_step0:{g['session_id']}:{g['item_id']}", g['span']['start'], g['span']['end'])] = g['span']['type']
    for l in read(handoff / 'labels_val.jsonl'):
        if not l['gold'] and l['end'] > l['start']:
            kinds[f"team_step0:{l['session_id']}:{l['item_id']}"].append((l['start'], l['end'], l['kind']))
    for s in read(handoff / 'sessions_val.jsonl'):
        groups[s['session_id']] = s['meta']['group_id']
    detector = MLSpanDetector(ROOT / args.checkpoint, device=args.device)
    threshold = detector.threshold if args.threshold is None else args.threshold
    gold_stats = defaultdict(Counter)   # (dimension, value) -> exact / boundary / missed
    fp_stats = defaultdict(Counter)
    for row in rows:
        session = row['id'].split(':')[1]
        src = origin(groups[session])
        channel = row['source']
        predicted = [(s['start'], s['end']) for s in detector.scan_text(row['text'], threshold=threshold)]
        gold = [(s['start'], s['end']) for s in row['spans']]
        for a, b in gold:
            outcome = 'exact' if (a, b) in predicted else 'boundary' if any(max(a, c) < min(b, d) for c, d in predicted) else 'missed'
            kind = gold_type.get((row['id'], a, b), 'unknown')
            for key in (('all', 'all'), ('type', kind), ('origin', src), ('channel', channel)):
                gold_stats[key][outcome] += 1
        for c, d in predicted:
            if (c, d) in gold:
                continue
            if any(max(a, c) < min(b, d) for a, b in gold):
                where = 'overlaps_gold'
            else:
                hit = [k for a, b, k in kinds.get(row['id'], []) if max(a, c) < min(b, d)]
                where = f'labelled_{hit[0]}' if hit else 'unlabelled'
            for key in (('all', 'all'), ('origin', src), ('channel', channel), ('where', where)):
                fp_stats[key]['fp'] += 1
    def table(stats):
        out = {}
        for (dim, value), c in sorted(stats.items()):
            out.setdefault(dim, {})[value] = dict(c)
        return out
    report = dict(model=args.model, checkpoint=args.checkpoint, threshold=threshold, rows=len(rows),
                  gold=table(gold_stats), false_positives=table(fp_stats),
                  notice='Validation only; counts by gold type, template origin, channel and labelled non-secret kind.')
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
