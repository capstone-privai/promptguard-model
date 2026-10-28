"""Refine from the released checkpoint with new code families and a fresh-family holdout."""
import argparse
import json
from pathlib import Path
from collections import Counter
from privai_model.io import write_json,write_jsonl
from privai_model.ml_span.generator import TEMPLATES,generate
from privai_model.ml_span.schema import load
from privai_model.ml_span.splitter import split,assert_disjoint

CODE_FAMILIES={row[0] for row in TEMPLATES if row[3].find('«V»')>=0 and row[2] in ('code','configuration','log','shell','tool_output') and row[0] in {
 'py_env_default','py_mapping','py_header','py_exception','js_template','js_process','go_config','java_prop',
 'sql_conn','terraform','docker_env','pem_block','raw_log_line','shell_config','jsonl_event','config_no_key'}}

def prepare(base_dir,new_data_dir,output,per_family=16,seed=20261027,exclusions=None):
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('refinement output must be new')
    old_train=load(Path(base_dir)/'train.jsonl')
    new=generate(per_family,seed,json.loads(Path(exclusions).read_text()) if exclusions else None,CODE_FAMILIES)
    fresh,manifest=split(new,seed=seed)
    new_dir=Path(new_data_dir)
    if new_dir.exists() and any(new_dir.iterdir()): raise FileExistsError('new data directory must be empty')
    new_dir.mkdir(parents=True,exist_ok=True)
    for name,rows in fresh.items(): write_jsonl(new_dir/(name+'.jsonl'),rows)
    combined={'train':old_train+fresh['train'],'validation':fresh['validation'],'test':fresh['test']}
    assert_disjoint(combined)
    output.mkdir(parents=True)
    for name,rows in combined.items(): write_jsonl(output/(name+'.jsonl'),rows)
    def stats(rows):
        return dict(records=len(rows),positive=sum(bool(r['spans']) for r in rows),negative=sum(not r['spans'] for r in rows),
            gold_spans=sum(len(r['spans']) for r in rows),families=len({r['family_id'] for r in rows}),
            sources=dict(Counter(r['source'] for r in rows)))
    manifest.update(schema_version='span-refinement-v1',notice='Base checkpoint has seen its earlier dataset; this is development refinement. New test templates/families are held out from this refinement.',
        added_code_families=sorted(CODE_FAMILIES),base_train_families=len({r['family_id'] for r in old_train}),
        composition={k:stats(v) for k,v in combined.items()},generation=dict(per_family=per_family,seed=seed,
            generated=len(new),public_exclusions=bool(exclusions)))
    write_json(output/'manifest.json',manifest)
    print(json.dumps(manifest['composition'],ensure_ascii=False))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-dir',default='data/span-synthetic-v1')
    p.add_argument('--new-data-dir',default='data/span-code-refinement-v1')
    p.add_argument('--output',default='data/span-refinement-v1');p.add_argument('--per-family',type=int,default=16)
    p.add_argument('--seed',type=int,default=20261027);p.add_argument('--exclusions',default='reports/ml_span/public-audit-exclusions.json')
    a=p.parse_args();prepare(a.base_dir,a.new_data_dir,a.output,a.per_family,a.seed,a.exclusions)
