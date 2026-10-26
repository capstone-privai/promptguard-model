"""Unicode code-point, end-exclusive offsets; validation never includes source text."""
import hashlib
import math
import re
from privai_model.data.adapter import iter_jsonl

FIELDS = ('family_id', 'template_id', 'origin', 'value_family')

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def canonical_template(row):
    text = row['text']
    for span in reversed(sorted(row['value_spans'], key=lambda s:s['start'])):
        text = text[:span['start']]+'<VALUE>'+text[span['end']:]
    return re.sub(r'\s+', ' ', text).strip()

def validate_spans(text, spans, scores=False):
    if not isinstance(text,str) or not isinstance(spans,list):
        raise ValueError('text and span list required')
    for span in spans:
        if not isinstance(span,dict): raise ValueError('span object required')
        a,b=span.get('start'),span.get('end')
        if type(a) is not int or type(b) is not int or not 0<=a<b<=len(text):
            raise ValueError('invalid character offsets')
        if span.get('secret_type','SECRET')!='SECRET': raise ValueError('unsupported secret type')
        if scores:
            s=span.get('score')
            if isinstance(s,bool) or not isinstance(s,(int,float)) or not math.isfinite(s) or not 0<=s<=1:
                raise ValueError('finite score in [0,1] required')

def validate(row):
    if not isinstance(row,dict) or row.get('schema_version')!='secret-span-v1':
        raise ValueError('invalid schema')
    if row.get('synthetic') is not True:
        raise ValueError('training accepts synthetic data only')
    for key in ('id','source','language',*FIELDS):
        if not isinstance(row.get(key),str) or not row[key]: raise ValueError('missing grouping metadata')
    validate_spans(row.get('text'),row.get('spans'))
    validate_spans(row['text'],row.get('value_spans'))
    for key in ('spans','value_spans'):
        bounds=sorted((s['start'],s['end']) for s in row[key])
        if any(a[1]>b[0] for a,b in zip(bounds,bounds[1:])):
            raise ValueError('overlapping annotations')
    if not {(s['start'],s['end']) for s in row['spans']} <= {(s['start'],s['end']) for s in row['value_spans']}:
        raise ValueError('gold spans must be registered value spans')
    return row

def load(path):
    rows=[validate(r) for r in iter_jsonl(path)]
    if not rows or len({r['id'] for r in rows})!=len(rows): raise ValueError('empty data or duplicate IDs')
    return rows

def group_keys(row):
    validate(row)
    return ([(k,row[k]) for k in FIELDS]+[('text',digest(row['text'])),
        ('canonical_template',digest(canonical_template(row)))]+
        [('value',digest(row['text'][s['start']:s['end']])) for s in row['value_spans']])
