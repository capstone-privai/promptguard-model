"""Copy a run's record (not its weights) between machines as one line of JSON.

  run_summary.py export RUN_DIR LOG [NAME=DELIMITER_REPORT ...]   prints the line
  run_summary.py import LINE_FILE DEST_DIR                        rebuilds the files under DEST_DIR

The line holds config.json, history.json, alignment.json, the bundle metadata, the training-progress
losses from the log and any delimiter-policy reports: settings and counts only, no text, values or
weights. Used to bring Colab runs into the pilot report without downloading checkpoints.
"""
import json
import sys
from pathlib import Path


def export(run, log, reports):
    run = Path(run)
    read = lambda p: json.loads((run / p).read_text())
    progress = []
    for line in Path(log).read_text(errors='replace').splitlines():
        if line.startswith('{"event": "training_progress"'):
            e = json.loads(line)
            progress.append([e['epoch'], e['batch'], e['batches'], e['mean_loss']])
    print(json.dumps(dict(name=run.name, config=read('config.json'), history=read('history.json'),
                          alignment=read('alignment.json'), bundle=read('model/span_bundle.json'), progress=progress,
                          reports={k: json.loads(Path(v).read_text()) for k, v in (r.split('=', 1) for r in reports)}),
                     separators=(',', ':')))


def restore(line_file, dest):
    d = json.loads(Path(line_file).read_text())
    out = Path(dest) / d['name']
    (out / 'model').mkdir(parents=True, exist_ok=True)
    for name in ('config', 'history', 'alignment'):
        (out / f'{name}.json').write_text(json.dumps(d[name], indent=2) + '\n')
    (out / 'model/span_bundle.json').write_text(json.dumps(d['bundle'], indent=2) + '\n')
    (out / 'progress.log').write_text(''.join(
        json.dumps(dict(event='training_progress', epoch=e, batch=b, batches=n, mean_loss=l)) + '\n' for e, b, n, l in d['progress']))
    for name, report in d['reports'].items():
        (out / f'{name}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    if sys.argv[1] == 'export':
        export(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        restore(sys.argv[2], sys.argv[3])
