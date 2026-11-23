"""Exact-span counts of span bundles on several labeled sets, each set on its own and each model's selection set
combined. Used to compare pilot follow-ups (e.g. B+CredData and B+v10) source by source on validation.

Usage: score_span_sources.py --model NAME=BUNDLE[:SET+SET] ... --set NAME=JSONL ... --device mps --output OUT.json
  `:SET+SET` names the sets the model was selected on; their union is reported as `own_validation`, which must
  match the training run's validation counts. Counts only; no text or values.
"""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from privai_model.ml_span.evaluation import measures
from privai_model.ml_span.runtime import MLSpanDetector
from privai_model.ml_span.schema import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', nargs='+', required=True)
    p.add_argument('--set', nargs='+', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS not available: stop (no CPU fallback)')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA not available: stop (no CPU fallback)')
    sets = dict(item.split('=', 1) for item in args.set)
    report = dict(device=args.device, sets=sets, models={})
    for item in args.model:
        name, rest = item.split('=', 1)
        path, _, own = rest.partition(':')
        own = own.split('+') if own else []
        if not set(own) <= set(sets):
            raise ValueError(f'{name}: selection sets {own} not among --set')
        detector = MLSpanDetector(ROOT / path, device=args.device)
        detector.delimiter_policy = None  # as trained
        entry = dict(bundle=path, threshold=detector.threshold, sets={})
        keys = {}
        for sname, spath in sets.items():
            gold, pred, start = set(), set(), time.time()
            for row in load(ROOT / spath):
                gold.update((sname, row['id'], s['start'], s['end']) for s in row['spans'])
                pred.update((sname, row['id'], s['start'], s['end']) for s in detector.scan_text(row['text']))
            keys[sname] = gold, pred
            entry['sets'][sname] = measures(gold, pred)
            print(json.dumps(dict(model=name, set=sname, seconds=round(time.time() - start),
                                  **{k: entry['sets'][sname][k] for k in ('tp', 'fp', 'fn', 'precision', 'f2')})), flush=True)
        if own:
            gold = set().union(*(keys[s][0] for s in own))
            pred = set().union(*(keys[s][1] for s in own))
            entry['own_validation'] = dict(sets=own, **measures(gold, pred))
            print(json.dumps(dict(model=name, own_validation=entry['own_validation'])), flush=True)
        report['models'][name] = entry
    Path(ROOT / args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
