"""Prepare independent synthetic data or train the isolated C-plan span model."""
import argparse
import json
from pathlib import Path
from collections import Counter
import yaml
from privai_model.io import write_json,write_jsonl
from privai_model.ml_span.generator import generate
from privai_model.ml_span.splitter import split


def prepare(output,per_family=12,exclusions=None,seed=713,include_families=None):
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('dataset output must be new')
    excluded=json.loads(Path(exclusions).read_text()) if exclusions else None
    rows=generate(per_family=per_family,seed=seed,exclusions=excluded,include_families=include_families)
    splits,manifest=split(rows)
    output.mkdir(parents=True)
    for name,items in splits.items(): write_jsonl(output/(name+'.jsonl'),items)
    manifest['composition']={name:dict(records=len(items),positive=sum(bool(r['spans']) for r in items),
        negative=sum(not r['spans'] for r in items),gold_spans=sum(len(r['spans']) for r in items),
        sources=dict(Counter(r['source'] for r in items)),families=len({r['family_id'] for r in items})) for name,items in splits.items()}
    expected=per_family*(sum(2 for row in __import__('privai_model.ml_span.generator',fromlist=['TEMPLATES']).TEMPLATES if not include_families or row[0] in include_families)+sum(1 for name,_ in __import__('privai_model.ml_span.generator',fromlist=['NEGATIVES']).NEGATIVES if not include_families or 'negative_'+name in include_families))
    manifest['generation']=dict(per_family=per_family,seed=seed,expected_before_filtering=expected,
        retained=len(rows),removed_by_exclusion_or_dedup=expected-len(rows),
        public_exclusion_scope=excluded.get('scope') if excluded else 'NOT CHECKED')
    write_json(output/'manifest.json',manifest)
    print(json.dumps(manifest['composition']))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True);a.add_argument('--per-family',type=int,default=12)
    a.add_argument('--exclusions');a.add_argument('--seed',type=int,default=713);a.add_argument('--families',nargs='*')
    a=sub.add_parser('train');a.add_argument('--config',required=True)
    args=p.parse_args()
    if args.command=='prepare': prepare(args.output,args.per_family,args.exclusions,args.seed,args.families)
    else:
        from privai_model.ml_span.training import train
        train(yaml.safe_load(Path(args.config).read_text()))
