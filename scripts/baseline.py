"""Run CredSweeper on existing manifest splits; feed files to the common evaluator."""
import argparse
import json
from pathlib import Path
from privai_model.baselines.credsweeper import CredSweeperBaseline
from privai_model.data.adapter import iter_jsonl,adapt_records
from privai_model.data.split import restore_split
from privai_model.io import write_json,write_jsonl

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--manifest')
    p.add_argument('--rules-only',action='store_true')
    p.add_argument('--splits',nargs='+',choices=['validation','test'],default=['validation','test'])
    a=p.parse_args()
    records=[r for r in iter_jsonl(a.data) if r.get('granularity')=='value']
    examples=adapt_records(records,require_label=bool(a.manifest))
    output=Path(a.output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('baseline output must be new')
    output.mkdir(parents=True,exist_ok=True)
    scanner=CredSweeperBaseline(use_ml=not a.rules_only)
    if a.manifest:
        splits=restore_split(examples,json.loads(Path(a.manifest).read_text()))
        by_id={r['candidate_id']:r for r in records}
        for name in a.splits:
            rows=scanner.predict([by_id[e.candidate_id] for e in splits[name]])
            write_jsonl(output/(name+'_predictions.jsonl'),rows)
    else:
        rows=scanner.predict(records)
        write_jsonl(output/'predictions.jsonl',rows)
    write_json(output/'baseline.json',scanner.metadata)
    print(json.dumps({'status':'completed','method':'CredSweeper','output':str(output)}))
