"""Evaluate frozen checkpoints on one hold-out and apply a decision rule written beforehand."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch
from privai_model.io import write_json
from privai_model.ml_span.schema import load
from privai_model.ml_span.evaluation import predict_rows
from privai_model.ml_span.rules import RuleSpanDetector

ROOT = Path(__file__).resolve().parents[1]


def compare_module():
    spec = importlib.util.spec_from_file_location('compare_span', ROOT / 'scripts/compare_span_checkpoints.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decide(result, decision):
    candidate = result[decision['candidate']['name']]['ml_only']
    baseline = result[decision['baseline']['name']]['ml_only']
    better = (candidate['f2'] or 0) > (baseline['f2'] or 0) and (candidate['precision'] or 0) >= (baseline['precision'] or 0)
    return dict(candidate_better=better, rule=decision['rule'], default_model=decision['default_model'],
                candidate={k: candidate[k] for k in ('tp', 'fp', 'fn', 'precision', 'recall', 'f2')},
                baseline={k: baseline[k] for k in ('tp', 'fp', 'fn', 'precision', 'recall', 'f2')})


def main(a):
    torch.set_num_threads(4)
    decision = json.loads(Path(a.decision).read_text())
    checkpoints = {decision['baseline']['name']: decision['baseline']['checkpoint'],
                   decision['candidate']['name']: decision['candidate']['checkpoint']}
    checkpoints.update({c['name']: c['checkpoint'] for c in decision.get('reference_only', [])})
    rows = load(Path(decision['dataset']))
    rules = RuleSpanDetector()
    rp, latency = predict_rows(rules, rows, True)
    result = compare_module().comparison(rows, rp, checkpoints)
    frozen = result[decision['candidate']['name']]['threshold']
    if frozen != decision['candidate']['threshold']:
        raise ValueError('candidate threshold differs from the predeclared one')
    report = dict(notice='Synthetic hold-out. Checkpoints and thresholds frozen before this run.',
                  decision_config=a.decision, dataset=decision['dataset'], rule_metadata=rules.metadata,
                  rule_latency=latency, result=result, decision=decide(result, decision))
    write_json(a.output, report)
    print(json.dumps(report['decision']))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--decision', required=True)
    p.add_argument('--output', required=True)
    main(p.parse_args())
