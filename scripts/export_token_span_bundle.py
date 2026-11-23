"""Export a token-span run (scripts/token_span.py train) as an ml-span bundle for MLSpanDetector.

  PYTHONPATH=src python scripts/export_token_span_bundle.py --run RUN_DIR --out BUNDLE_DIR \
      [--dtype bfloat16] [--set key=value] [--check-validation --device cuda]

RUN_DIR is a training output folder (config.json, selection.json, run.json, epoch-N/). The epoch and cutoff are the
ones selection.json chose on validation; there is no option to pick others. --set overrides keys of the run's
config.json, e.g. model_name=runs/backbones/... with local_files_only=true, or data_dir=... for the check.
--check-validation scores validation with the exported bundle (MLSpanDetector) and records the counts in
span_bundle.json. Counts only; no text or values are printed.
"""
import argparse
import json

import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'])
    p.add_argument('--set', action='append', default=[])
    p.add_argument('--check-validation', action='store_true')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    from privai_model.token_span.export import check_validation, export, read_run
    overrides = {}
    for item in args.set:
        key, value = item.split('=', 1)
        overrides[key] = yaml.safe_load(value)
    meta = export(args.run, args.out, dtype=args.dtype, overrides=overrides)
    summary = dict(out=args.out, epoch=meta['best_epoch'], threshold=meta['threshold'], dtype=meta['torch_dtype'],
                   bidirectional=meta['bidirectional'], validation=meta['validation'])
    if args.check_validation:
        config, _, _ = read_run(args.run)
        summary['check'] = check_validation(args.out, overrides.get('data_dir', config['data_dir']), args.device)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
