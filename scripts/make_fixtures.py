import argparse
from pathlib import Path
from privai_model.data.synthetic import hard_cases
from privai_model.io import write_jsonl

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='data/synthetic/hard-cases.jsonl')
    p.add_argument('--long-context',action='store_true')
    a=p.parse_args()
    path=Path(a.output)
    if path.exists():
        raise FileExistsError('fixture output exists; select a new path')
    path.parent.mkdir(parents=True,exist_ok=True)
    rows=hard_cases(a.long_context)
    write_jsonl(path,rows)
    print(f'Wrote {len(rows)} fabricated candidates; not performance evaluation data')
