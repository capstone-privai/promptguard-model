"""Does the verifier in the real detector mask what the policy comparison said? Counts only.

  PYTHONPATH=src python scripts/check_verifier_in_detector.py --detector ../promptguard-claude-demoV0 \
      --bundle runs/verifier/creddata-mmbert-small/model [--data runs/data/span-creddata-traj-v1] \
      [--split validation] [--expected runs/verifier/creddata-mmbert-small/system_validation.json]

Runs the PromptGuard detector checkout with candidate_policy "verifier" (promptguard-claude-demoV0, --policy verifier)
on one split's span rows and scores the spans it masks against the gold spans, the way
scripts/evaluate_verifier_system.py scores "strong + verifier" on the prepared candidates. Built like load_detector
builds it: the checkout's detector.json adapters and its "verifier" entry, with this repository as the model
repository and --bundle as the bundle. With --expected (that script's output), prints whether the counts match.

The candidates match when the checkout's adapters are the ones that made the verifier data (manifest.json of
prepare_verifier_candidates.py records the commit and versions). Validation only: test stays closed until the run is
frozen (scripts/final_evaluate.py).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from privai_model.ml_span.schema import load  # noqa: E402
from privai_model.verifier.candidates import code_points, masked_span_metrics  # noqa: E402

FIELDS = ('masked', 'false_masks', 'gold_spans', 'gold_reached', 'precision', 'recall', 'f2')


def build_detector(detector_root, bundle, device):
    sys.path.insert(0, str(detector_root / 'src' / 'detector'))
    os.environ['PG_DETECTOR_MODE'] = 'normal'
    os.environ['PG_DETECTOR_CANDIDATE_POLICY'] = 'majority'  # adapters as configured; the verifier is added below
    from pgdetector import Detector, load_detector
    from pgdetector.ensemble import DEFAULT_CONFIG
    from pgdetector.verifier import VerifierStage
    base = load_detector()
    config = Path(os.environ.get('PG_DETECTOR_CONFIG') or DEFAULT_CONFIG).resolve()  # the file load_detector read
    options = json.loads(config.read_text(encoding='utf-8')).get('verifier', {})
    stage = VerifierStage({**options, 'repository': str(ROOT), 'bundle': str(bundle), 'device': device}, config.parent)
    return Detector(base.adapters, base.policy, base.skipped, sorted(base.candidate_only), 'verifier', stage)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--detector', required=True, help='promptguard-claude-demoV0 checkout with --policy verifier')
    p.add_argument('--bundle', required=True, help='verifier bundle folder (bundle.json and weights)')
    p.add_argument('--data', default=str(ROOT / 'runs/data/span-creddata-traj-v1'))
    p.add_argument('--split', default='validation', choices=('train', 'validation'))
    p.add_argument('--expected', help="evaluate_verifier_system.py output for the same split and bundle")
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    detector = build_detector(Path(args.detector).resolve(), Path(args.bundle).resolve(), args.device)
    described = detector.describe()
    rows = load(Path(args.data) / f'{args.split}.jsonl')
    masked, errors = {}, 0
    began = time.perf_counter()
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        for row, result in zip(chunk, detector.scan_many([(r['text'], r['source']) for r in chunk])):
            errors += bool(result['errors'])  # codes only; a failed scan masks every candidate (fail closed)
            to_cp = code_points(row['text'])
            masked[row['id']] = [(to_cp(f['start']), to_cp(f['end'])) for f in result['findings']]
    gold = {r['id']: [(s['start'], s['end']) for s in r['spans']] for r in rows}
    metrics = masked_span_metrics(masked, gold)
    report = dict(split=args.split, rows=len(rows), rows_with_errors=errors,
                  seconds=round(time.perf_counter() - began, 1), verifier=described['verifier'],
                  adapters=described['adapters'], skipped=described['skipped'], strong_verifier=metrics)
    if args.expected:
        expected = json.loads(Path(args.expected).read_text(encoding='utf-8'))['strong_verifier']
        report['expected'] = {k: expected[k] for k in FIELDS}
        report['match'] = all(metrics[k] == expected[k] for k in FIELDS)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
