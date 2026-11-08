"""Collect the step-0 pilot into reports/ml_span/pilot-step0.json: data, settings, validation history, gates.

Usage: report_pilot_step0.py RUN=DIR ... (RUN is e.g. A-seed110; DIR holds config.json, history.json,
alignment.json and model/span_bundle.json). Reads the data manifests and the gate reports under
runs/pilot-reports. Validation and counts only: the team's test sets are not scored here.
"""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ('model_name', 'backbone_revision', 'seed', 'epochs', 'patience', 'batch_size', 'length_grouped_batches',
            'learning_rate', 'weight_decay', 'class_weights', 'max_length', 'stride', 'thresholds', 'shared_key_kinds')


def read(path):
    path = ROOT / path
    return json.loads(path.read_text()) if path.exists() else None


def short(m):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in ('tp', 'fp', 'fn', 'precision', 'recall', 'f2')}


def trim(q):
    """A delimiter-policy report: own-validation gate, then development sets (before/after, fixed, broken)."""
    if not q:
        return None
    return dict(policy=q['policy'], eligible=q['eligible'], validation_before=short(q['validation']['before']),
                validation_after=short(q['validation']['after']), fixed=q['validation']['fixed'], broken=q['validation']['broken'],
                gold_preservation=q.get('gold_preservation'),
                development={k: dict(before=short(v['before']), after=short(v['after']), fixed=v['fixed'], broken=v['broken'])
                             for k, v in q['development'].items()})


def main():
    runs = {}
    for spec in sys.argv[1:]:
        name, folder = spec.split('=', 1)
        config, history = read(f'{folder}/config.json'), read(f'{folder}/history.json')
        bundle = read(f'{folder}/model/span_bundle.json')
        model = name.split('-')[0].lower()
        runs[name] = dict(
            data=f'runs/data/span-pilot-step0-{model}', device=config.get('device'),
            gpu=(config.get('colab') or {}).get('gpu', 'Apple MPS' if config.get('device') == 'mps' else None),
            commit=(config.get('colab') or {}).get('commit'), settings={k: config.get(k) for k in SETTINGS},
            alignment=read(f'{folder}/alignment.json'),
            history=[dict(epoch=h['epoch'], loss=round(h['loss'], 6), cutoff=h['threshold'], **short(h['validation'])) for h in history],
            selected_epoch=bundle['best_epoch'], cutoff=bundle['threshold'],
            quote_trim=trim(read(f'runs/pilot-reports/delimiters-pilot-{name.lower()}.json')),
            quote_trim_v2=trim(read(f'runs/pilot-reports/delimiters-v2-pilot-{name.lower()}.json')))
    data = {}
    for m in ('a', 'b', 'bhn'):
        manifest = read(f'runs/data/span-pilot-step0-{m}/manifest.json')
        if manifest:
            data[m.upper()] = {k: manifest[k] for k in ('team_sessions', 'composition', 'shared_keys', 'fingerprint',
                                                        'team_sha256', 'base', 'added') if k in manifest}
    baselines = {}
    for name in ('v9', 'v10'):
        for prefix in ('delimiters', 'delimiters-v2'):
            q = read(f'runs/pilot-reports/{prefix}-{name}.json')
            if q:
                baselines.setdefault(name, {})[q['policy']] = trim(q)
    report = dict(
        notice=('Team step-0 pilot (promptguard-data ml-train-pilot 5b812c7). Model A: hand + privesc; model B: hand + '
                'LLM + privesc. Same recipe; only the train and val files differ, and each seed is a separate run. '
                'B+HN (follow-up) is B plus our secret-free hard negatives, selected on B val. Validation only: the team '
                'scores its test sets first. Counts and settings only; no text or values.'),
        data=data, runs=runs, quote_trim_on_published_bundles=baselines,
        diagnostics={p.stem: read(f'runs/pilot-reports/{p.name}') for p in sorted((ROOT / 'runs/pilot-reports').glob('diag-*.json'))})
    out = ROOT / 'reports/ml_span/pilot-step0.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(out, len(out.read_text()))


if __name__ == '__main__':
    main()
