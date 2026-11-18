"""Build the back-end verifier's data: the PromptGuard detector's candidates on span rows, as candidate-v1.

  PYTHONPATH=src python scripts/prepare_verifier_candidates.py --detector ../promptguard-claude-demoV0 \
      [--data runs/data/span-creddata-traj-v1] [--out runs/data/verifier-creddata-traj-v1]

The detector runs as configured in its checkout (src/detector/detector.json, with PG_DETECTOR_ENABLE / _DISABLE
honoured), so the candidates are the ones PromptGuard would produce. CredSweeper must be installed (the detector
requires it); gitleaks is used when its binary is on PATH and skipped otherwise. The adapters that ran are recorded.

Writes, under --out (values and context: never commit):
  all/candidates.jsonl.gz, all/split_manifest.json              every candidate, strong and ambiguous
  ambiguous/candidates.jsonl.gz, ambiguous/split_manifest.json  only the low-confidence ones the verifier will see
  gold_spans.json     gold offsets per row (no values), for the system-level evaluation
  analysis.json       false-positive analysis on train and validation, counts only (test is a hold-out)
  manifest.json       detector adapters and versions, its git commit, source data fingerprint, counts
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from privai_model.io import write_json  # noqa: E402
from privai_model.ml_span.schema import load  # noqa: E402
from privai_model.verifier.candidates import SPLITS, analysis, records_for_row, scan_rows, split_manifest  # noqa: E402


def write_gz(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def git_commit(path):
    try:
        return subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--detector', required=True, help='promptguard-claude-demoV0 checkout')
    p.add_argument('--data', default=str(ROOT / 'runs/data/span-creddata-traj-v1'))
    p.add_argument('--out', default=str(ROOT / 'runs/data/verifier-creddata-traj-v1'))
    p.add_argument('--context-lines', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    data, out, detector_root = Path(args.data), Path(args.out), Path(args.detector).resolve()
    if out.exists():
        raise FileExistsError(out)
    sys.path.insert(0, str(detector_root / 'src' / 'detector'))
    os.environ.setdefault('PG_DETECTOR_MODE', 'normal')
    from pgdetector import load_detector
    detector = load_detector()
    described = detector.describe()
    print(json.dumps(dict(adapters=described['adapters'], skipped=described['skipped'],
                          candidate_policy=described['candidate_policy'], consensus=described['consensus'])))

    rows = {name: load(data / f'{name}.jsonl') for name in SPLITS}
    records = []
    for name in SPLITS:
        began = time.perf_counter()
        for row, candidates in scan_rows(detector, rows[name], args.batch_size):
            records.extend(records_for_row(row, name, candidates, args.context_lines))
        print(json.dumps(dict(split=name, rows=len(rows[name]), seconds=round(time.perf_counter() - began, 1),
                              candidates=sum(r['source']['split'] == name for r in records))))

    counts = {}
    for tier, kept in (('all', records), ('ambiguous', [r for r in records if r['detector']['strength'] == 'ambiguous'])):
        write_gz(out / tier / 'candidates.jsonl.gz', kept)
        manifest = split_manifest(kept)
        write_json(out / tier / 'split_manifest.json', manifest)
        counts[tier] = manifest['statistics']['splits']
    write_json(out / 'gold_spans.json', {name: {r['id']: [[s['start'], s['end']] for s in r['spans']] for r in rows[name]}
                                         for name in SPLITS})
    write_json(out / 'analysis.json', analysis(records, rows))
    source_manifest = data / 'manifest.json'
    write_json(out / 'manifest.json', dict(
        detector=dict(adapters=described['adapters'], skipped=described['skipped'], consensus=described['consensus'],
                      candidate_policy=described['candidate_policy'], commit=git_commit(detector_root),
                      env={k: v for k, v in os.environ.items() if k.startswith('PG_DETECTOR')}),
        source=dict(path=str(data), fingerprint=json.loads(source_manifest.read_text(encoding='utf-8')).get('fingerprint')
                    if source_manifest.exists() else None),
        context_lines=args.context_lines, labels='MASK when the candidate overlaps a gold span, else KEEP',
        counts=counts, tiers={f'{s}/{t}': n for (s, t), n in
                              sorted(Counter((r['source']['split'], r['detector']['strength']) for r in records).items())},
        notice='Values and context of realistic fake (and possibly real) credentials: keep under runs/, never commit.'))
    print(json.dumps(dict(counts=counts), ensure_ascii=False))


if __name__ == '__main__':
    main()
