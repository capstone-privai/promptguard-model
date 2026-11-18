"""Train or test a BIO token-classification span detector (src/privai_model/token_span/train.py).

  PYTHONPATH=src python scripts/token_span.py train --config configs/token-span-creddata-qwen.yaml [--set key=value] [--resume]
  PYTHONPATH=src python scripts/token_span.py test --config ... --weights OUTPUT/epoch-N --threshold T --split test

`test` is for after selection: it scores the frozen epoch and cutoff on one split and prints counts only.
"""
import argparse
import json
from pathlib import Path

import yaml


def load_config(path, sets):
    config = yaml.safe_load(Path(path).read_text())
    for item in sets:
        key, value = item.split('=', 1)
        config[key] = yaml.safe_load(value)
    return config


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--config', required=True)
    t.add_argument('--set', action='append', default=[])
    t.add_argument('--resume', action='store_true')
    e = sub.add_parser('test')
    e.add_argument('--config', required=True)
    e.add_argument('--set', action='append', default=[])
    e.add_argument('--weights', required=True)
    e.add_argument('--threshold', type=float, required=True)
    e.add_argument('--split', required=True)
    args = p.parse_args()
    config = load_config(args.config, args.set)
    from privai_model.token_span import train as ts
    if args.command == 'train':
        selection = ts.train(config, resume=args.resume)
        print(json.dumps(dict(selected=selection['selected'])))
        return
    from privai_model.ml_span.evaluation import measures
    from privai_model.ml_span.schema import load
    device = ts.resolve_device(config.get('device', 'auto'))
    rows = load(Path(config['data_dir']) / f'{args.split}.jsonl')
    tokenizer = ts.load_tokenizer(config)
    model = ts.place(ts.load_model(dict(config, gradient_checkpointing=False), adapter=args.weights), device)
    predictions = ts.predict(model, tokenizer, rows, config['max_length'], config['stride'], device,
                             ts.amp_dtype(config, device), config.get('eval_batch_size', 16))
    gold = ts.gold_keys(rows)
    kept = {(rid, s['start'], s['end']) for rid, spans in predictions.items() for s in spans if s['score'] >= args.threshold}
    print(json.dumps(dict(split=args.split, rows=len(rows), weights=args.weights, threshold=args.threshold,
                          exact=measures(gold, kept))))


if __name__ == '__main__':
    main()
