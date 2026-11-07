"""Measure the opt-in closing-quote trim (quote-delimiters-v1) on unchanged weights.

Each checkpoint is judged on its own validation set (exact-span F2 must rise and precision must not
fall); every other set is development evidence. The trim never changes a gold span in data/ or in the
pilot and v11 splits, which the report checks too. Counts only; no text or values.
"""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from privai_model.ml_span.delimiters import POLICY, trim_quote_delimiters
from privai_model.ml_span.evaluation import measures
from privai_model.ml_span.runtime import MLSpanDetector
from privai_model.ml_span.schema import load


def compare(detector, rows):
    gold, before, after = set(), set(), set()
    for row in rows:
        key = lambda s: (row['id'], s['start'], s['end'])
        gold.update(key(s) for s in row['spans'])
        prediction = detector.scan_text(row['text'])
        before.update(key(s) for s in prediction)
        after.update(key(s) for s in trim_quote_delimiters(row['text'], prediction))
    return dict(rows=len(rows), before=measures(gold, before), after=measures(gold, after),
                fixed=len((after - before) & gold), broken=len((before - after) & gold))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--validation', required=True, help='the checkpoint\'s own validation JSONL')
    p.add_argument('--development', nargs='*', default=[], help='NAME=PATH sets reported only')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    detector = MLSpanDetector(ROOT / args.checkpoint, device=args.device)
    detector.delimiter_policy = None  # baseline: identical weights, cutoff and other filters
    report = dict(policy=POLICY, checkpoint=args.checkpoint, device=args.device, threshold=detector.threshold,
                  validation=dict(path=args.validation, **compare(detector, load(ROOT / args.validation))), development={})
    for item in args.development:
        name, path = item.split('=', 1)
        report['development'][name] = compare(detector, load(ROOT / path))
    v = report['validation']
    report['eligible'] = (v['after']['f2'] or 0) > (v['before']['f2'] or 0) and \
        (v['after']['precision'] or 0) >= (v['before']['precision'] or 0)
    paths = sorted((ROOT / 'data').glob('span-*/*.jsonl')) + sorted((ROOT / 'runs/data').glob('span-*/*.jsonl'))
    checked = changed = 0
    for path in paths:
        for row in load(path):
            out = trim_quote_delimiters(row['text'], row['spans'])
            checked += len(row['spans'])
            changed += sum((a['start'], a['end']) != (b['start'], b['end']) for a, b in zip(out, row['spans']))
    report['gold_preservation'] = dict(spans_checked=checked, changed=changed)
    Path(ROOT / args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(eligible=report['eligible'], validation={k: v[k] for k in ('fixed', 'broken')},
                          before=v['before'], after=v['after'], gold_preservation=report['gold_preservation'])))


if __name__ == '__main__':
    main()
