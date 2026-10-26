"""Connected components across ALL leakage keys, not concatenated group IDs."""
import json
from types import SimpleNamespace
from privai_model.data.split import split_groups
from .schema import group_keys, digest


def assert_disjoint(splits):
    seen={}
    for name,rows in splits.items():
        for row in rows:
            for key in group_keys(row):
                if key in seen and seen[key]!=name: raise ValueError('split leakage detected')
                seen[key]=name


def split(rows,seed=42,ratios=(.8,.1,.1)):
    if not rows or len({r['id'] for r in rows})!=len(rows): raise ValueError('empty data or duplicate IDs')
    parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,row in enumerate(rows):
        for key in group_keys(row):
            if key in seen: parent[root(i)]=root(seen[key])
            seen[key]=i
    component={}
    for i,row in enumerate(rows): component.setdefault(root(i),[]).append(row['id'])
    ids={k:min(v) for k,v in component.items()}
    proxies=[SimpleNamespace(candidate_id=r['id'],group_id=ids[root(i)],label=int(bool(r['spans'])),
        type=r['source']) for i,r in enumerate(rows)]
    grouped,manifest=split_groups(proxies,seed,ratios)
    by_id={r['id']:r for r in rows}
    result={k:[by_id[e.candidate_id] for e in v] for k,v in grouped.items()}
    assert_disjoint(result)
    manifest['dataset_fingerprint']=digest(json.dumps(sorted(rows,key=lambda r:r['id']),sort_keys=True,ensure_ascii=False))
    manifest['schema_version']='span-connected-split-v1'
    return result,manifest
