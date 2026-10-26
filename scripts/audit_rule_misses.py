"""Read public examples in memory; persist only metadata and exclusion hashes."""
import argparse
import hashlib
import json
import re
from collections import Counter
from urllib.request import urlopen
from privai_model.io import write_json

BASE = 'https://baseballtour.duckdns.org/promptguard/'
def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def template(text, spans):
    for a, b in sorted(set(spans), reverse=True):
        text = text[:a] + '<VALUE>' + text[b:]
    return re.sub(r'\s+', ' ', text).strip()

def audit(output, selected_commit=None):
    def get(path):
        with urlopen(BASE+path, timeout=60) as f:
            return json.load(f)
    index = get('data/index.json')
    commit = next((c for c in index['commits'] if c['sha']==selected_commit), None) if selected_commit else index['commits'][-1]
    if commit is None: raise RuntimeError('requested commit unavailable')
    if commit['detection']['state'] != 'done':
        raise RuntimeError('latest result unavailable; explicit snapshot selection required')
    key = commit['detection']['key']
    summary = get(f'data/results/detection/{key}/summary.json')
    result = dict(source=BASE+'#o=missed', result_key=key, commit=commit['sha'],
                  evaluated_at=summary['evaluated_at'], system=summary['system'],
                  benchmark=summary['benchmark'], totals=summary['totals'], datasets=summary['datasets'])
    cases, seen, values, templates = [], set(), set(), set()
    loaded, schemas = {}, {}
    for dataset in summary['datasets']:
        stem = dataset['key']
        payload = get(f'data/results/detection/{key}/examples/{stem}.json')
        loaded[stem] = len(payload['examples'])
        schemas[stem] = {k:v for k,v in payload.items() if k!='examples'}
        for ex in payload['examples']:
            identity = (stem, ex['id'])
            if identity in seen:
                continue
            seen.add(identity)
            text = ex['text']
            # UI highlight() consumes these as excerpt-relative Unicode offsets.
            spans = [(g[0],g[1]) for g in ex['gold'] if 0<=g[0]<g[1]<=len(text)]
            values.update(digest(text[a:b]) for a,b in spans)
            templates.add(digest(template(text,spans)))
            for gi,g in enumerate(ex['gold']):
                a,b,kind,status = g[:4]
                if status!='missed':
                    continue
                valid = type(a) is int and type(b) is int and 0<=a<b<=len(text)
                overlaps = valid and any(max(a,e[0])<min(b,e[1]) for e in ex['edits'])
                value = text[a:b] if valid else ''
                context = text[:a]+text[b:] if valid else ''
                facts = []
                if '\n' in value or '\r' in value: facts.append('multiline_value')
                if '\\n' in value: facts.append('escaped_newline_value')
                if re.search('[가-힣]',context) and re.search('[A-Za-z]',context): facts.append('mixed_ko_en_context')
                if len(ex['gold'])>1: facts.append('multiple_gold_spans')
                if ex['channel']=='prompt' or ex.get('group')=='자연어': facts.append('prompt_or_natural_language_source')
                cases.append(dict(dataset=stem,id=ex['id'],gold_index=gi,channel=ex['channel'],
                    group=ex.get('group'),type=kind,start=a,end=b,source_offset=ex['offset'],
                    source_length=ex['length'],excerpt_length=len(text),edit_count=len(ex['edits']),
                    gold_count=len(ex['gold']),facts=facts,
                    mapping_status='invalid_bounds' if not valid else 'status_overlap_conflict' if overlaps else 'consistent_no_overlap',
                    source_complete=ex['offset']==0 and len(text)==ex['length'],
                    root_cause='UNDETERMINED: per-detector candidates/filter trace unavailable'))
    result.update(loaded_examples=loaded,example_metadata=schemas,missed_span_count=len(cases),
        missed_example_count=len({(c['dataset'],c['id']) for c in cases}),
        by_dataset=dict(Counter(c['dataset'] for c in cases)),
        by_type=dict(Counter(c['type'] for c in cases)),
        mapping=dict(Counter(c['mapping_status'] for c in cases)),
        observed_facts=dict(Counter(f for c in cases for f in c['facts'])),cases=cases)
    write_json(output,result)
    write_json(str(output).replace('.json','-exclusions.json'),dict(value_sha256=sorted(values),template_sha256=sorted(templates),
        scope='downloaded public example samples only; full original hold-out unavailable'))
    print(json.dumps({k:result[k] for k in ('result_key','missed_span_count','missed_example_count','by_dataset','by_type','mapping','observed_facts')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='reports/ml_span/public-audit.json')
    p.add_argument('--commit');a=p.parse_args()
    audit(a.output,a.commit)
