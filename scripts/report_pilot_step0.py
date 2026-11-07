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
            quote_trim=read(f'runs/pilot-reports/delimiters-pilot-{name.lower()}.json'))
        if runs[name]['quote_trim']:
            q = runs[name]['quote_trim']
            runs[name]['quote_trim'] = dict(eligible=q['eligible'], validation_before=short(q['validation']['before']),
                                            validation_after=short(q['validation']['after']), fixed=q['validation']['fixed'],
                                            broken=q['validation']['broken'],
                                            development={k: dict(fixed=v['fixed'], broken=v['broken']) for k, v in q['development'].items()})
    data = {}
    for m in 'ab':
        manifest = read(f'runs/data/span-pilot-step0-{m}/manifest.json')
        data[m.upper()] = {k: manifest[k] for k in ('team_sessions', 'composition', 'shared_keys', 'fingerprint', 'team_sha256')}
    baselines = {}
    for name in ('v9', 'v10'):
        q = read(f'runs/pilot-reports/delimiters-{name}.json')
        if q:
            baselines[name] = dict(own_validation_fixed=q['validation']['fixed'], eligible=q['eligible'],
                                   development={k: dict(fixed=v['fixed'], broken=v['broken'], f2_before=round(v['before']['f2'] or 0, 4),
                                                        f2_after=round(v['after']['f2'] or 0, 4)) for k, v in q['development'].items()})
    report = dict(
        notice=('Team step-0 pilot (promptguard-data ml-train-pilot 5b812c7). Model A: hand + privesc; model B: hand + '
                'LLM + privesc. Same recipe and seed; only the train and val files differ. Validation only: the team '
                'scores its test sets first. Counts and settings only; no text or values.'),
        data=data, runs=runs, quote_trim_on_published_bundles=baselines,
        diagnostics={p.stem: read(f'runs/pilot-reports/{p.name}') for p in sorted((ROOT / 'runs/pilot-reports').glob('diag-*.json'))})
    out = ROOT / 'reports/ml_span/pilot-step0.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(out, len(out.read_text()))


if __name__ == '__main__':
    main()
