"""Unofficial: score val predictions with the team's metric-detection scoring rules.

The team's scorer (promptguard-data metric-detection/scripts/evaluate.py) refuses train and val sessions on
purpose, so this does not run it. It imports only its scoring rules (evaluation/scorer/span_scoring.py and
report/aggregate.py's F2) and applies them to the model's val spans, each span taken as one masking edit.
PromptGuard itself is not in the loop (no other detectors, no placeholder re-masking of repeated values).

Prints counts only: no text, values or IDs.

  PYTHONPATH=src .venv/bin/python scripts/score_val_team_rules.py --team runs/team-data/promptguard-data/metric-detection \
    --rows runs/data/span-pilot-step0-a/validation.jsonl --mmbert models/span-pilot-step0-a-seed110
  ... --spans qwen-a-e3-spans.json --cutoff 0.957716187327652
"""
import argparse
import json
import sys
from pathlib import Path


def team_rules(team):
    sys.path.insert(0, str(Path(team).resolve()))
    from evaluation.dataset.schema import GoldSpan
    from evaluation.report.aggregate import f2_score
    from evaluation.scorer.edits import Edit, apply_edits
    from evaluation.scorer.span_scoring import SpanStatus, score_item
    return GoldSpan, f2_score, Edit, apply_edits, SpanStatus, score_item


def merged(spans):
    """Overlapping spans become one edit (the scorer rejects overlapping edits, as PromptGuard merges them)."""
    out = []
    for start, end in sorted(spans):
        if out and start < out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def ours(rows, predicted):
    gold = {(r['id'], s['start'], s['end']) for r in rows for s in r['spans']}
    pred = {(rid, a, b) for rid, spans in predicted.items() for a, b in spans}
    tp, fp, fn = len(gold & pred), len(pred - gold), len(gold - pred)
    return dict(tp=tp, fp=fp, fn=fn, f2=5 * tp / (5 * tp + fp + 4 * fn) if tp + fp + fn else None)


def team(rows, predicted, rules):
    GoldSpan, f2_score, Edit, apply_edits, SpanStatus, score_item = rules
    statuses, edits_total, within, overlapping, outside = [], 0, 0, 0, 0
    for row in rows:
        edits = [Edit(a, b, '[SECRET]') for a, b in merged(predicted.get(row['id'], []))]
        gold = [GoldSpan(0, i, s['start'], s['end'], 'SECRET') for i, s in enumerate(row['spans'])]
        item = score_item(row['id'], 0, row['source'], row['text'], apply_edits(row['text'], edits), edits, gold)
        statuses += [g.status for g in item.gold_results]
        for e in item.edit_results:
            edits_total += 1
            within += e.within_gold
            overlapping += e.overlaps_gold and not e.within_gold
            outside += not e.overlaps_gold
    recall = statuses.count(SpanStatus.EXACT) / len(statuses)
    precision = within / edits_total if edits_total else None
    return dict(gold=len(statuses), **{s.value: statuses.count(s) for s in SpanStatus},
                edits=edits_total, edits_within_gold=within, edits_spilling_over_gold=overlapping,
                edits_outside_gold=outside, recall=recall, precision=precision, f2=f2_score(precision, recall))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--team', required=True, help='promptguard-data metric-detection folder')
    p.add_argument('--rows', required=True)
    p.add_argument('--mmbert', help='mmBERT bundle folder; its own cutoff is used')
    p.add_argument('--delimiter-policy', choices=['quote-delimiters-v1', 'quote-delimiters-v2'])
    p.add_argument('--spans', help='JSON {row id: [[start, end, score], ...]} from another detector')
    p.add_argument('--cutoff', type=float, help='cutoff for --spans')
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    rows = [json.loads(line) for line in open(args.rows, encoding='utf-8')]
    if args.mmbert:
        from privai_model.ml_span.runtime import MLSpanDetector
        detector = MLSpanDetector(args.mmbert, device=args.device, delimiter_policy=args.delimiter_policy)
        cutoff = detector.threshold
        predicted = {r['id']: [(s['start'], s['end']) for s in detector.scan_text(r['text'], threshold=cutoff)] for r in rows}
        source = dict(model=args.mmbert, cutoff=cutoff, delimiter_policy=args.delimiter_policy, device=str(detector.device))
    else:
        spans = json.load(open(args.spans, encoding='utf-8'))
        if not set(spans) <= {r['id'] for r in rows}:
            raise SystemExit('span IDs not in the rows')
        predicted = {rid: [(a, b) for a, b, score in v if score >= args.cutoff] for rid, v in spans.items()}
        source = dict(spans=args.spans, cutoff=args.cutoff)
    print(json.dumps(dict(source=source, rows=len(rows), ours=ours(rows, predicted),
                          team_rules=team(rows, predicted, team_rules(args.team)))))


if __name__ == '__main__':
    main()
