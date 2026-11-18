"""What PromptGuard would mask with and without the back-end verifier, at span level, on one split.

  PYTHONPATH=src python scripts/evaluate_verifier_system.py --data runs/data/verifier-creddata-traj-v1 \
      --predictions RUN/validation_predictions.jsonl --threshold T --split validation

Compares three policies on the same candidates (counts only):
  union             mask every candidate (today's candidate_policy "union")
  strong_only       mask strong candidates, drop every ambiguous one (no verifier)
  strong+verifier   mask strong candidates and the ambiguous ones the verifier scores at or above T
Recall is the share of gold spans overlapped by a masked candidate (gold no candidate reaches stays missed);
precision is the share of masked candidates that overlap a gold span. Use validation to choose; test once, frozen.
"""
import argparse
import gzip
import json
from pathlib import Path

from privai_model.verifier.candidates import system_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='output folder of prepare_verifier_candidates.py')
    p.add_argument('--predictions', required=True, help='prediction-v1 rows with candidate_id and p_mask')
    p.add_argument('--threshold', type=float, required=True)
    p.add_argument('--split', required=True)
    args = p.parse_args()
    data = Path(args.data)
    with gzip.open(data / 'all' / 'candidates.jsonl.gz', 'rt', encoding='utf-8') as f:
        records = [r for r in map(json.loads, f) if r['source']['split'] == args.split]
    gold = {rid: [tuple(s) for s in spans] for rid, spans in json.loads((data / 'gold_spans.json').read_text(encoding='utf-8'))[args.split].items()}
    scores = {}
    with open(args.predictions, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            scores[row['candidate_id']] = row['p_mask']
    ambiguous = {r['candidate_id'] for r in records if r['detector']['strength'] == 'ambiguous'}
    if not ambiguous <= set(scores):
        raise SystemExit(f'{len(ambiguous - set(scores))} ambiguous candidates have no verifier score')
    print(json.dumps(dict(
        split=args.split, threshold=args.threshold, candidates=len(records), ambiguous=len(ambiguous),
        union=system_metrics(records, gold),
        strong_only=system_metrics(records, gold, scores={}, threshold=2.0),
        strong_verifier=system_metrics(records, gold, scores=scores, threshold=args.threshold)), ensure_ascii=False))


if __name__ == '__main__':
    main()
