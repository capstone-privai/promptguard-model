"""Compare unchanged v9 weights before/after the opt-in candidate policy.

Reports contain counts and offsets only; no source text or credential values.
Historical datasets are development regression evidence, not a new holdout.
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
from privai_model.ml_span.function_words import POLICY, filter_function_words
from privai_model.ml_span.schema import load
from privai_model.ml_span.evaluation import measures
from privai_model.qwen_span.data import EVAL_SETS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/span-detector-v9')
    p.add_argument('--output', default='reports/ml_span/v9-function-word-fix.json')
    args = p.parse_args()
    torch.set_num_threads(4)
    detector = MLSpanDetector(ROOT / args.checkpoint)
    assert detector.candidate_filter == POLICY
    detector.candidate_filter = None  # raw pre-change baseline, with identical weights/threshold
    report = dict(policy=POLICY, checkpoint=args.checkpoint, threshold=detector.threshold,
                  notice='Historical regression data; not independent holdout. Runtime candidate filter, no retraining.',
                  datasets={})
    total_removed_tp = 0
    for name, relative in EVAL_SETS.items():
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError('Expected regression dataset missing: ' + relative)
        rows = load(path)
        gold, before, after = set(), set(), set()
        for row in rows:
            key = lambda span: (row['id'], span['start'], span['end'])
            gold.update(key(s) for s in row['spans'])
            prediction = detector.scan_text(row['text'])
            before.update(key(s) for s in prediction)
            after.update(key(s) for s in filter_function_words(row['text'], prediction))
        removed_tp = len((before - after) & gold)
        total_removed_tp += removed_tp
        result = dict(rows=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      before=measures(gold, before), after=measures(gold, after),
                      removed_tp=removed_tp, removed_fp=len((before-after)-gold))
        report['datasets'][name] = result
        print(name, result['before'], result['after'], flush=True)
    checked, suppressed_gold = 0, 0
    for path in sorted((ROOT / 'data').glob('span-*/*.jsonl')):
        for row in load(path):
            gold = row['spans']
            checked += len(gold)
            suppressed_gold += len(gold) - len(filter_function_words(row['text'], gold))
    report['gold_preservation'] = dict(spans_checked=checked, suppressed=suppressed_gold)
    report['passed'] = total_removed_tp == 0 and suppressed_gold == 0
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise SystemExit('Regression: candidate policy suppresses a labelled secret')
    print('Passed: no known gold spans or baseline true positives removed', flush=True)


if __name__ == '__main__':
    main()
