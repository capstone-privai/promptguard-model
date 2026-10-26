"""Prepare independent synthetic data or train the isolated C-plan span model."""
import argparse
import json
from pathlib import Path
from collections import Counter
import yaml
from privai_model.io import write_json,write_jsonl
from privai_model.ml_span.generator import generate
from privai_model.ml_span.splitter import split


def prepare(output,per_family=12,exclusions=None):
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('dataset output must be new')
    excluded=json.loads(Path(exclusions).read_text()) if exclusions else None
    rows=generate(per_family=per_family,exclusions=excluded)
    splits,manifest=split(rows)
    output.mkdir(parents=True)
    for name,items in splits.items(): write_jsonl(output/(name+'.jsonl'),items)
    manifest['composition']={name:dict(records=len(items),positive=sum(bool(r['spans']) for r in items),
        negative=sum(not r['spans'] for r in items),gold_spans=sum(len(r['spans']) for r in items),
        sources=dict(Counter(r['source'] for r in items)),families=len({r['family_id'] for r in items})) for name,items in splits.items()}
    manifest['generation']=dict(per_family=per_family,seed=713,expected_before_filtering=71*per_family,
        retained=len(rows),removed_by_exclusion_or_dedup=71*per_family-len(rows),
        public_exclusion_scope=excluded.get('scope') if excluded else 'NOT CHECKED')
    write_json(output/'manifest.json',manifest)
    print(json.dumps(manifest['composition']))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True);a.add_argument('--per-family',type=int,default=12)
    a.add_argument('--exclusions')
    a=sub.add_parser('train');a.add_argument('--config',required=True)
    args=p.parse_args()
    if args.command=='prepare': prepare(args.output,args.per_family,args.exclusions)
    else:
        from privai_model.ml_span.training import train
        train(yaml.safe_load(Path(args.config).read_text()))
