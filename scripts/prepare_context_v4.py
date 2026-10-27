"""Create a fixed fresh-family challenge split with historical evaluation exclusion."""
import json
from pathlib import Path
from collections import Counter
from privai_model.io import write_json,write_jsonl
from privai_model.ml_span.context_data import generate_context
from privai_model.ml_span.schema import load,digest
from privai_model.ml_span.splitter import assert_disjoint

def main():
    output=Path('data/span-context-v4')
    if output.exists(): raise FileExistsError(output)
    exclusions=json.loads(Path('reports/ml_span/public-audit-exclusions.json').read_text())
    splits,manifest=generate_context(exclusions=exclusions)
    splits['train']=load(Path('data/span-refinement-v3/train.jsonl'))+splits['train']
    assert_disjoint(splits)
    previous=[]
    for directory in ['data/span-synthetic-v1','data/span-refinement-v1','data/span-refinement-v3']:
        for split in ['validation','test']: previous.extend(load(Path(directory)/(split+'.jsonl')))
    assert_disjoint({'train':splits['train'],'previous_evaluation':previous})
    output.mkdir()
    for split,rows in splits.items(): write_jsonl(output/(split+'.jsonl'),rows)
    manifest.update(composition={k:dict(records=len(v),gold_spans=sum(len(r['spans']) for r in v),families=len({r['family_id'] for r in v}),
        positive=sum(bool(r['spans']) for r in v),sources=dict(Counter(r['source'] for r in v))) for k,v in splits.items()},
        fingerprint=digest(json.dumps(splits,sort_keys=True,ensure_ascii=False)),
        notice='Fresh independent synthetic families. All prior validation/test families excluded from train across entire checkpoint lineage. Fixed assignment before training. Public page value/template hash exclusions applied, not semantic similarity proof.')
    write_json(output/'manifest.json',manifest)
    print(json.dumps(manifest['composition']))
if __name__=='__main__': main()
