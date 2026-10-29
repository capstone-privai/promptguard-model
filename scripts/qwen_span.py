"""Qwen generative span-detector experiment: prepare -> smoke -> train -> predict -> select -> report.

Selection (adapter epoch and confidence cutoff) uses v9 validation only and is frozen in
selection.json before any other dataset is scored. Reports hold counts/IDs/offsets only.

Run from the repository root, in a normal Terminal on the Mac (MPS is hidden in sandboxes):
    .venv/bin/python scripts/qwen_span.py prepare
    .venv/bin/python scripts/qwen_span.py smoke --config configs/qwen-span-coder-0.5b.yaml
    .venv/bin/python scripts/qwen_span.py train --config configs/qwen-span-coder-0.5b.yaml
See docs/ml_span/QWEN_SPAN.md for the full sequence.
"""
import argparse
import json
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from privai_model.io import write_json  # noqa: E402
from privai_model.ml_span.schema import load  # noqa: E402
from privai_model.qwen_span import data as qdata  # noqa: E402
from privai_model.qwen_span import evaluation as qeval  # noqa: E402

PREDICTIONS = Path('runs/qwen-span/predictions')
REPORTS = Path('reports/qwen_span')
V9_REPORT = Path('reports/ml_span/context-v9-comparison.json')


def load_config(path, overrides):
    config = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split('=', 1)
        if key not in config:
            raise SystemExit(f'unknown config key: {key}')
        config[key] = yaml.safe_load(value)
    return config


def cmd_prepare(args):
    manifest = qdata.prepare(ROOT, args.sft_dir, extra_train=args.extra_train)
    write_json(REPORTS / f'sft-manifest{args.run}.json', manifest)
    for name, r in manifest['roundtrip'].items():
        print(f"{name}: {r['recovered_exactly']}/{r['gold_spans']} gold spans recovered exactly")
    for name, r in manifest['leakage'].items():
        print(f"leakage {name}: {r['rows_sharing_any_key']} rows share a group/text/template/value key with train")


def cmd_smoke(args):
    import torch
    from privai_model.qwen_span.modeling import load_model, load_tokenizer, encode_example, completion_loss, reference_loss
    from privai_model.qwen_span.train import train, read_sft, text_of
    from privai_model.qwen_span.infer import QwenSpanDetector
    config = load_config(args.config, args.set)
    tokenizer = load_tokenizer(config['model_name'], config.get('revision'), config.get('local_files_only', False))
    model = load_model(config['model_name'], 'float32', config.get('revision'), config.get('local_files_only', False))
    records = read_sft(Path(config['sft_dir']) / 'train.jsonl')
    with torch.no_grad():
        for record in records[:3]:
            e = encode_example(tokenizer, text_of(record), record['messages'][2]['content'])
            ids = torch.tensor([e['input_ids']])
            a = completion_loss(model, ids, e['prompt_length']).item()
            b = reference_loss(model, ids, e['prompt_length']).item()
            if abs(a - b) > 1e-4 * max(1.0, abs(b)):
                raise SystemExit(f'completion loss mismatch: {a} vs {b}')
            decoded = tokenizer.decode(e['input_ids'][e['prompt_length']:], skip_special_tokens=True)
            if decoded != record['messages'][2]['content']:
                raise SystemExit('completion tokens do not decode back to the target answer')
    print('completion-only loss equals masked full loss; targets decode exactly')
    del model
    smoke = dict(config, output_dir=str(Path(config['output_dir']).with_name('smoke')), limit=16, epochs=1,
                 gradient_accumulation=4, save_every=2, log_every=1)
    out = train(smoke)
    detector = QwenSpanDetector(config['model_name'], out / 'epoch-1', config.get('device', 'auto'),
                                config.get('dtype', 'float32'), config.get('revision'), config.get('local_files_only', False),
                                config.get('max_new_tokens', 512))
    rows = load(ROOT / qdata.EVAL_SETS['validation'])[:4]
    for row in rows:
        d = detector.scan_detail(row['text'])
        print(json.dumps(dict(id=row['id'], gold=len(row['spans']), predicted=len(d['spans']), error=d['error'],
                              counts=d['counts'], ms=round(d['latency_ms']))))
    print('smoke passed: train, save, reload and generate all ran; quality is not judged here')


def cmd_train(args):
    from privai_model.qwen_span.train import train
    config = load_config(args.config, args.set)
    print('adapters in', train(config, resume=args.resume))


def cmd_predict(args):
    from privai_model.qwen_span.infer import QwenSpanDetector
    config = load_config(args.config, args.set)
    adapter = None if args.adapter == 'base' else args.adapter
    detector = QwenSpanDetector(config['model_name'], adapter, config.get('device', 'auto'), config.get('dtype', 'float32'),
                                config.get('revision'), config.get('local_files_only', False), config.get('max_new_tokens', 512))
    names = list(qdata.EVAL_SETS) if args.sets == 'all' else args.sets.split(',')
    for name in names:
        rows = load(ROOT / qdata.EVAL_SETS[name])
        if args.limit:
            rows = rows[:args.limit]
        path = PREDICTIONS / args.tag / f'{name}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        done = set(qeval.read_predictions(path)) if path.exists() else set()
        with path.open('a', encoding='utf-8') as f:
            for i, row in enumerate(rows):
                if row['id'] in done:
                    continue
                d = detector.scan_detail(row['text'])
                f.write(json.dumps(dict(id=row['id'], **d)) + '\n')
                f.flush()
                if (i + 1) % 25 == 0:
                    print(f'{args.tag}/{name}: {i + 1}/{len(rows)}', flush=True)
        write_json(PREDICTIONS / args.tag / 'meta.json', dict(model_name=config['model_name'], adapter=args.adapter,
            revision=config.get('revision'), dtype=config.get('dtype'), max_new_tokens=config.get('max_new_tokens', 512)))
        print(f'{args.tag}/{name}: complete')


def cmd_bench(args):
    """Latency only, under v9's measurement condition (CPU, fixed threads, sequential, warm model).
    Nothing is scored or stored except timings, so it can run on any set after selection is frozen."""
    import torch
    from privai_model.qwen_span.infer import QwenSpanDetector
    config = load_config(args.config, args.set)
    torch.set_num_threads(args.threads)
    adapter = None if args.adapter == 'base' else args.adapter
    detector = QwenSpanDetector(config['model_name'], adapter, args.device, config.get('dtype', 'float32'),
                                config.get('revision'), config.get('local_files_only', False), config.get('max_new_tokens', 512))
    rows = load(ROOT / qdata.EVAL_SETS[args.dataset])[:args.limit]
    detector.scan_detail(rows[0]['text'])  # warm-up, excluded
    timings = [detector.scan_detail(r['text']) for r in rows]
    result = dict(adapter=args.adapter, dataset=args.dataset, device=args.device, threads=args.threads,
                  dtype=config.get('dtype'), records=len(rows),
                  latency=qeval.latency([t['latency_ms'] for t in timings]),
                  generated_tokens=qeval.latency([t['generated_tokens'] for t in timings]),
                  includes='tokenization, greedy generation, offset recovery; sequential warm model; warm-up call excluded')
    write_json(REPORTS / f'latency-{args.tag}-{args.device}.json', result)
    print(json.dumps(result['latency']))


def cmd_v9predict(args):
    """Run the v9 bundle live (CPU, 4 threads, its own threshold) on sets missing from the saved v9 report.
    Checks first that the live run reproduces the stored validation counts."""
    import torch
    from privai_model.ml_span.runtime import MLSpanDetector
    torch.set_num_threads(4)
    detector = MLSpanDetector(str(ROOT / 'models/span-detector-v9'), 'cpu')
    v9_report = json.loads(V9_REPORT.read_text())['datasets']
    for name in ['validation'] + args.sets.split(','):
        rows = load(ROOT / qdata.EVAL_SETS[name])
        out = PREDICTIONS / 'v9' / f'{name}.jsonl'
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', encoding='utf-8') as f:
            for row in rows:
                began = __import__('time').perf_counter()
                spans = detector.scan_text(row['text'])
                ms = (__import__('time').perf_counter() - began) * 1000
                f.write(json.dumps(dict(id=row['id'], spans=[dict(start=s['start'], end=s['end'], score=s['score']) for s in spans],
                                        error=None, counts={}, latency_ms=ms)) + '\n')
        predictions = qeval.read_predictions(out)
        if name in v9_report:
            live = qeval.predicted_keys(rows, predictions, 0.0)
            if live != qeval.v9_predictions(rows, v9_report[name]):
                raise SystemExit(f'live v9 run does not reproduce stored {name} predictions')
            print(f'v9 live run reproduces stored {name} predictions exactly')
        else:
            print(name, json.dumps(qeval.score(rows, predictions, 0.0)['exact']))


def scored(tag, name, rows):
    predictions = qeval.read_predictions(PREDICTIONS / tag / f'{name}.jsonl')
    if set(predictions) != {r['id'] for r in rows}:
        return None
    return predictions


def cmd_select(args):
    """Freeze adapter and cutoff on validation before any other set is scored."""
    if (REPORTS / f'selection{args.run}.json').exists() and not args.force:
        raise SystemExit('selection.json already frozen; pass --force only if no test set has been looked at')
    rows = load(ROOT / qdata.EVAL_SETS['validation'])
    v9 = json.loads(V9_REPORT.read_text())['datasets']['validation']['candidate']['ml_only']
    candidates = {}
    for tag in args.tags.split(','):
        predictions = scored(tag, 'validation', rows)
        if predictions is None:
            raise SystemExit(f'{tag}: validation predictions incomplete')
        raw = qeval.score(rows, predictions, 0.0)
        calibrated = qeval.select_threshold(rows, predictions, args.min_precision)
        candidates[tag] = dict(raw=raw, calibration=calibrated)
    def rank(tag):
        s = candidates[tag]['calibration']['selected']
        return (candidates[tag]['calibration']['eligible'], s['f2'] or 0, s['precision'] or 0)
    selected = max(candidates, key=rank)
    selection = dict(policy=('Validation only. Per adapter: cutoff maximizing exact F2 over all observed confidences '
                             '(precision tie-break), restricted to precision >= min_precision when given. Adapter with '
                             'highest eligible F2 is selected. Frozen before scoring any other dataset.'),
                     min_precision=args.min_precision, v9_validation=v9, candidates=candidates,
                     selected=dict(tag=selected, threshold=candidates[selected]['calibration']['selected']['threshold']))
    write_json(REPORTS / f'selection{args.run}.json', selection)
    for tag, c in candidates.items():
        s = c['calibration']['selected']; r = c['raw']['exact']
        print(f"{tag}: raw {r['tp']}/{r['fp']}/{r['fn']}  cutoff {s['threshold']:.4f} -> {s['tp']}/{s['fp']}/{s['fn']} "
              f"F2 {s['f2']:.4f} eligible={c['calibration']['eligible']}  invalid={c['raw']['invalid_outputs']}")
    print(f"selected {selected} at {selection['selected']['threshold']:.6f}; v9 validation {v9['tp']}/{v9['fp']}/{v9['fn']}")


def cmd_report(args):
    selection = json.loads((REPORTS / f'selection{args.run}.json').read_text())
    tag, threshold = selection['selected']['tag'], selection['selected']['threshold']
    v9_report = json.loads(V9_REPORT.read_text())['datasets']
    tags = [tag] + [t for t in (args.also or '').split(',') if t]
    report = dict(notice=('Synthetic only. Adapter and cutoff frozen on validation (selection.json). '
                          '*_development sets were used while developing v4-v9 and are development checks. '
                          'v9 numbers are rebuilt from reports/ml_span/context-v9-comparison.json.'),
                  selection=selection['selected'], development_after=args.development_note, datasets={})
    for name, path in qdata.EVAL_SETS.items():
        rows = load(ROOT / path)
        if name in v9_report:
            v9 = qeval.v9_predictions(rows, v9_report[name])
        else:
            live = scored('v9', name, rows)
            if live is None:
                continue
            v9 = {(k, s['start'], s['end']) for k, p in live.items() for s in p['spans']}
        entry = {}
        for t in tags:
            predictions = scored(t, name, rows)
            if predictions is None:
                continue
            cut = threshold if t == tag else 0.0
            entry[t] = dict(score=qeval.score(rows, predictions, cut), raw=qeval.score(rows, predictions, 0.0)['exact'],
                            vs_v9=qeval.compare_with_v9(rows, predictions, cut, v9),
                            by_family=qeval.breakdown(rows, predictions, cut, 'family_id'),
                            by_source=qeval.breakdown(rows, predictions, cut, 'source'),
                            by_kind=qeval.breakdown(rows, predictions, cut, 'kind'),
                            errors=qeval.errors(rows, predictions, cut))
        if entry:
            report['datasets'][name] = entry
    if args.with_rules:
        report['rule_union'] = rule_union(tag, threshold)
    write_json(REPORTS / f'comparison{args.run}.json', report)
    print('| Dataset | v9 TP/FP/FN | Qwen TP/FP/FN | Qwen-only TP | Qwen new FP | v9∪Qwen TP/FP/FN | invalid |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for name, entry in report['datasets'].items():
        if tag not in entry:
            continue
        c = entry[tag]['vs_v9']; u = c['ml_union_v9_or_qwen']
        print(f"| {name} | {c['v9']['tp']}/{c['v9']['fp']}/{c['v9']['fn']} | {c['qwen']['tp']}/{c['qwen']['fp']}/{c['qwen']['fn']} "
              f"| {c['qwen_only_tp']} | {c['qwen_new_fp']} | {u['tp']}/{u['fp']}/{u['fn']} | {sum(entry[tag]['score']['invalid_outputs'].values())} |")


def rule_union(tag, threshold):
    """Same Rule+ML candidate union as v9 (requires the `baselines` extra)."""
    from privai_model.ml_span.rules import RuleSpanDetector
    from privai_model.ml_span.evaluation import evaluate, predict_rows
    rules = RuleSpanDetector()
    result = dict(rule_metadata=rules.metadata, datasets={})
    for name, path in qdata.EVAL_SETS.items():
        rows = load(ROOT / path)
        predictions = scored(tag, name, rows)
        if predictions is None:
            continue
        rp, _ = predict_rows(rules, rows, True)
        ml = {k: [dict(start=s['start'], end=s['end'], secret_type='SECRET', detector='qwen', score=s['score'])
                  for s in v['spans']] for k, v in predictions.items()}
        result['datasets'][name] = evaluate(rows, rp, ml, threshold)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('prepare'); s.add_argument('--sft-dir', default='runs/qwen-span/sft')
    s.add_argument('--extra-train', action='append', default=[], help='additional train JSONL (repo-relative)')
    s.add_argument('--run', default='', help='report suffix, e.g. -v2'); s.set_defaults(func=cmd_prepare)
    s = sub.add_parser('v9predict'); s.add_argument('--sets', required=True); s.set_defaults(func=cmd_v9predict)
    for name, func in [('smoke', cmd_smoke), ('train', cmd_train), ('predict', cmd_predict)]:
        s = sub.add_parser(name); s.add_argument('--config', required=True)
        s.add_argument('--set', action='append', default=[], metavar='KEY=VALUE'); s.set_defaults(func=func)
        if name == 'train':
            s.add_argument('--resume', action='store_true')
        if name == 'predict':
            s.add_argument('--adapter', required=True, help="adapter directory, or 'base' for zero-shot")
            s.add_argument('--tag', required=True); s.add_argument('--sets', default='validation')
            s.add_argument('--limit', type=int)
    s = sub.add_parser('bench'); s.add_argument('--config', required=True)
    s.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    s.add_argument('--adapter', required=True); s.add_argument('--tag', required=True)
    s.add_argument('--dataset', default='fresh_test'); s.add_argument('--limit', type=int, default=96)
    s.add_argument('--device', default='cpu'); s.add_argument('--threads', type=int, default=4); s.set_defaults(func=cmd_bench)
    s = sub.add_parser('select'); s.add_argument('--tags', required=True)
    s.add_argument('--min-precision', type=float); s.add_argument('--force', action='store_true')
    s.add_argument('--run', default='', help='selection file suffix, e.g. -v2'); s.set_defaults(func=cmd_select)
    s = sub.add_parser('report'); s.add_argument('--also', help='extra tags (e.g. zero-shot) reported at cutoff 0')
    s.add_argument('--with-rules', action='store_true'); s.add_argument('--run', default='')
    s.add_argument('--development-note', help='sets whose results were seen before this run was designed')
    s.set_defaults(func=cmd_report)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
