"""Scan or mask text with the bundled Qwen span detector (models/qwen-span-v3).

Input is one of --text, --file, --jsonl ({"id", "text"} per line) or stdin. Output is JSON
with offsets and scores only; --mask adds the masked text. Secret values are never printed.

Exit code: 0 clean, 1 secrets found, 2 scan failed (fail closed: treat as not clean).

    .venv/bin/python scripts/qwen_scan.py --text 'token = "..."'
    .venv/bin/python scripts/qwen_scan.py --file prompt.txt --mask
    .venv/bin/python scripts/qwen_scan.py --jsonl inputs.jsonl --output results.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from privai_model.qwen_span.wrapper import DEFAULT_BUNDLE, QwenSecretScanner  # noqa: E402

CLEAN, FOUND, FAILED = 0, 1, 2


def handle(scanner, text, mask, placeholder):
    if mask:
        r = scanner.mask(text, placeholder)
        out = dict(status='error' if r.status == 'blocked' else 'ok', mask_status=r.status, error=r.error,
                   findings=[f.__dict__ for f in r.findings], masked_text=r.text)
        code = FAILED if r.status == 'blocked' else FOUND if r.findings else CLEAN
    else:
        r = scanner.scan(text)
        out = r.to_dict()
        code = FAILED if r.status != 'ok' else FOUND if r.findings else CLEAN
    return out, code


def records(args):
    if args.jsonl:
        for n, line in enumerate(Path(args.jsonl).read_text(encoding='utf-8').splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row.get('text'), str):
                    raise SystemExit(f'line {n}: "text" string required')
                yield row.get('id', n), row['text']
    elif args.text is not None:
        yield None, args.text
    elif args.file:
        yield None, Path(args.file).read_text(encoding='utf-8')
    else:
        yield None, sys.stdin.read()


def main(argv=None, scanner=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--text'); source.add_argument('--file'); source.add_argument('--jsonl')
    p.add_argument('--mask', action='store_true', help='also return the text with findings replaced')
    p.add_argument('--placeholder', default='[SECRET]')
    p.add_argument('--bundle', default=str(DEFAULT_BUNDLE))
    p.add_argument('--adapter', help='override the bundle adapter directory')
    p.add_argument('--threshold', type=float, help='override the validation-selected cutoff')
    p.add_argument('--device', default='auto', help='auto, mps, cuda or cpu')
    p.add_argument('--local-files-only', action='store_true', help='do not contact Hugging Face')
    p.add_argument('--output', help='write results here instead of stdout')
    args = p.parse_args(argv)
    scanner = scanner or QwenSecretScanner(args.bundle, adapter=args.adapter, threshold=args.threshold,
                                           device=args.device, local_files_only=args.local_files_only)
    lines, worst = [], CLEAN
    for rid, text in records(args):
        out, code = handle(scanner, text, args.mask, args.placeholder)
        if rid is not None:
            out = dict(id=rid, **out)
        lines.append(json.dumps(out, ensure_ascii=False))
        worst = max(worst, code)
    body = '\n'.join(lines) + '\n'
    if args.output:
        Path(args.output).write_text(body, encoding='utf-8')
    else:
        sys.stdout.write(body)
    return worst


if __name__ == '__main__':
    sys.exit(main())
