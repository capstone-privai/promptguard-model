"""Evaluate any method using independent validation/test candidate-ID score files."""
import argparse
import json
from pathlib import Path
from privai_model.data.adapter import load_examples, iter_jsonl
from privai_model.data.split import restore_split
from privai_model.evaluation.evaluator import select_thresholds, evaluate
from privai_model.io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--validation-predictions', required=True)
    parser.add_argument('--test-predictions', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    splits = restore_split(load_examples(args.data), json.loads(Path(args.manifest).read_text()))
    validation = list(iter_jsonl(args.validation_predictions))
    thresholds = select_thresholds(splits['validation'], validation)
    # Thresholds are finalized before test prediction data is opened.
    test = list(iter_jsonl(args.test_predictions))
    report = {'threshold_source': 'validation', 'thresholds': thresholds,
              'notice': 'Provisional data is not final PrivAI performance. Hard predictions have only two score levels.',
              'validation': {k: evaluate(splits['validation'], validation, t) for k,t in thresholds.items()},
              'test': {k: evaluate(splits['test'], test, t) for k,t in thresholds.items()}}
    write_json(args.output, report)
    print(json.dumps({'thresholds': thresholds, 'output': args.output}))


if __name__ == '__main__':
    main()
