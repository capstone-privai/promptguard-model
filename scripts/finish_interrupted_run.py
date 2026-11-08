"""Write span_bundle.json for a training run that stopped before it finished (e.g. a Colab runtime reset).

Training saves the weights at every epoch that raises validation F2 and rewrites history.json after each
epoch, but writes the bundle only at the end. This writes the bundle the run would have had if it had
stopped after its last completed epoch: the saved weights are the best completed epoch's, and the cutoff
is the one validation chose for that epoch. The bundle records that the run was interrupted.

Usage: finish_interrupted_run.py RUN_DIR
"""
import json
import sys
from pathlib import Path
from privai_model.io import write_json
from privai_model.ml_span.bio import LABELS


def finish(run):
    run = Path(run)
    bundle = run / 'model/span_bundle.json'
    if bundle.exists():
        raise FileExistsError(bundle)
    if not (run / 'model/model.safetensors').is_file():
        raise FileNotFoundError(f'no saved weights in {run}/model')
    config = json.loads((run / 'config.json').read_text())
    history = json.loads((run / 'history.json').read_text())
    best = None
    for h in history:  # the training rule: the first epoch with the highest F2
        if best is None or (h['validation']['f2'] or 0.) > (best['validation']['f2'] or 0.):
            best = h
    write_json(bundle, dict(
        schema_version='ml-span-bundle-v1', max_length=config['max_length'], stride=config['stride'],
        threshold=best['threshold'], threshold_source='synthetic validation F2',
        backbone=config.get('backbone_id', config['model_name']), revision=config.get('backbone_revision', config.get('revision')),
        label_mapping=LABELS, best_epoch=best['epoch'], production_ready=False,
        interrupted=dict(completed_epochs=len(history), epochs=config['epochs'],
                         note='training stopped early; bundle written by scripts/finish_interrupted_run.py')))
    return best


if __name__ == '__main__':
    best = finish(sys.argv[1])
    print(json.dumps(dict(best_epoch=best['epoch'], threshold=best['threshold'], f2=best['validation']['f2'])))
