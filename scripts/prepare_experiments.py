"""Prepare, never execute, a shared-split validation-only experiment matrix."""
import argparse
import json
from pathlib import Path
import shlex
import yaml
from privai_model.data.adapter import load_examples
from privai_model.data.split import split_groups
from privai_model.io import write_json


def prepare(data,output,seed=42,windows=(64,128,256,512,1024)):
    root=Path(__file__).resolve().parents[1]
    data=Path(data).resolve();output=Path(output).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError('experiment plan directory must be new')
    examples=load_examples(data)
    _,manifest=split_groups(examples,seed)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'split_manifest.json',manifest)
    plans=[]
    for backbone in ('ettin68m','ettin150m','modernbert-base','mmbert-small'):
        base=yaml.safe_load((root/'configs'/f'{backbone}.yaml').read_text())
        for window in windows:
            for feature in (False,True):
                name=f'{backbone}-w{window}-features-{int(feature)}'
                config=dict(base,data_path=str(data),output_dir=str(output/'runs'/name),seed=seed,
                            split_manifest=str(output/'split_manifest.json'),window_size=window,
                            character_features=feature,model_version=name,evaluate_test=False)
                path=output/'configs'/f'{name}.yaml';path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(yaml.safe_dump(config,sort_keys=False))
                plans.append({'name':name,'config':str(path),'evaluate_test':False})
    write_json(output/'plan.json',{'notice':'Run only after real labels/group definitions are approved; select using validation only.',
                                  'data_fingerprint':manifest['dataset_fingerprint'],'experiments':plans})
    commands=['#!/usr/bin/env bash','set -euo pipefail',f'cd {shlex.quote(str(root))}']
    commands.extend('uv run python scripts/train.py --config '+shlex.quote(p['config']) for p in plans)
    (output/'run_validation_matrix.sh').write_text('\n'.join(commands)+'\n')
    return plans


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--windows',nargs='+',type=int,default=[64,128,256,512,1024])
    a=p.parse_args();plans=prepare(a.data,a.output,a.seed,a.windows)
    print(json.dumps({'configs_created':len(plans),'training_executed':False,'test_evaluation_enabled':False}))
