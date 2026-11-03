"""Measure the opt-in minimum value length guard on unchanged published weights.

Validation decides (exact-span F2 must not fall, precision must not fall); the other sets are historical
development evidence. Reports contain counts only; no source text or credential values.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from privai_model.ml_span.runtime import MLSpanDetector
from privai_model.ml_span.short_values import PROMPTGUARD_MIN_VALUE_LENGTH, filter_short_values
from privai_model.ml_span.schema import load
from privai_model.ml_span.evaluation import measures
from privai_model.qwen_span.data import EVAL_SETS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', default=['models/span-detector-v9', 'models/span-detector-v10'])
    p.add_argument('--min-value-length', type=int, default=PROMPTGUARD_MIN_VALUE_LENGTH)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default='reports/ml_span/min-value-length.json')
    args = p.parse_args()
    torch.set_num_threads(4)
    report = dict(min_value_length=args.min_value_length, device=args.device, decision_set='validation',
                  notice='Runtime candidate filter, no retraining. Only validation decides; other sets are development.',
                  checkpoints={})
    for checkpoint in args.checkpoints:
        detector = MLSpanDetector(ROOT / checkpoint, device=args.device)
        detector.min_value_length = None  # baseline: identical weights, threshold and other filters
        result = dict(threshold=detector.threshold, datasets={})
        for name, relative in EVAL_SETS.items():
            path = ROOT / relative
            rows = load(path)
            gold, before, after = set(), set(), set()
            for row in rows:
                key = lambda span: (row['id'], span['start'], span['end'])
                gold.update(key(s) for s in row['spans'])
                prediction = detector.scan_text(row['text'])
                before.update(key(s) for s in prediction)
                after.update(key(s) for s in filter_short_values(row['text'], prediction, args.min_value_length))
            result['datasets'][name] = dict(rows=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                            before=measures(gold, before), after=measures(gold, after),
                                            removed_tp=len((before - after) & gold), removed_fp=len((before - after) - gold))
            print(checkpoint, name, result['datasets'][name]['removed_tp'], result['datasets'][name]['removed_fp'], flush=True)
        v = result['datasets']['validation']
        result['validation_passed'] = (v['after']['f2'] or 0) >= (v['before']['f2'] or 0) and \
            (v['after']['precision'] or 0) >= (v['before']['precision'] or 0)
        result['removed_tp_any_set'] = sum(d['removed_tp'] for d in result['datasets'].values())
        report['checkpoints'][checkpoint] = result
    checked = suppressed = 0
    for path in sorted((ROOT / 'data').glob('span-*/*.jsonl')):
        for row in load(path):
            checked += len(row['spans'])
            suppressed += len(row['spans']) - len(filter_short_values(row['text'], row['spans'], args.min_value_length))
    report['gold_preservation'] = dict(spans_checked=checked, suppressed=suppressed)
    out = ROOT / args.output
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({c: dict(validation_passed=r['validation_passed'], removed_tp_any_set=r['removed_tp_any_set'])
                      for c, r in report['checkpoints'].items()} | dict(gold_preservation=report['gold_preservation'])))


if __name__ == '__main__':
    main()
