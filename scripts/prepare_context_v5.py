"""Fresh training values plus label-preserving context paraphrases; fixed v4 holdout."""
import json
import re
from pathlib import Path
from collections import Counter
from privai_model.io import write_json,write_jsonl
from privai_model.ml_span.context_data import generate_context
from privai_model.ml_span.schema import validate,load,digest,canonical_template
from privai_model.ml_span.splitter import assert_disjoint

REPLACEMENTS={
 'password':'passphrase','Password':'Passphrase','PASSWORD':'PASSPHRASE',
 'secret':'credential','Secret':'Credential','SECRET':'CREDENTIAL',
 'authentication':'login','Authentication':'Login',
 'request':'correlation','Request':'Correlation','REQUEST':'CORRELATION',
 'trace':'tracking','reference':'identifier','worker':'scheduler',
 '비밀번호':'접속 암호','암호':'비밀번호','비밀 문자열':'접속용 비밀값',
}
PATTERN=re.compile('|'.join(re.escape(k) for k in sorted(REPLACEMENTS,key=len,reverse=True)))


def paraphrase(row,index):
    # Replace context only, never touch generated credential/public values.
    text='';spans=[];value_spans=[];cursor=0
    gold={(s['start'],s['end']) for s in row['spans']}
    for span in row['value_spans']:
        text+=PATTERN.sub(lambda m:REPLACEMENTS[m[0]],row['text'][cursor:span['start']])
        start=len(text);text+=row['text'][span['start']:span['end']]
        new=dict(start=start,end=len(text),secret_type='SECRET');value_spans.append(new)
        if (span['start'],span['end']) in gold:spans.append(new.copy())
        cursor=span['end']
    text+=PATTERN.sub(lambda m:REPLACEMENTS[m[0]],row['text'][cursor:])
    return validate(dict(row,id='paraphrase5:'+row['id'],text=text,spans=spans,value_spans=value_spans))


def main():
    output=Path('data/span-context-v5')
    if output.exists():raise FileExistsError(output)
    exclusions=json.loads(Path('reports/ml_span/public-audit-exclusions.json').read_text())
    fresh,_=generate_context(per_family=24,seed=20261031,exclusions=exclusions)
    added=[paraphrase(row,i) for i,row in enumerate(fresh['train'])]
    added=[r for r in added if digest(canonical_template(r)) not in exclusions['template_sha256']]
    splits={name:load(Path('data/span-context-v4')/(name+'.jsonl')) for name in ('train','validation','test')}
    splits['train']+=added
    assert_disjoint(splits)
    prior=[]
    for folder in ['data/span-synthetic-v1','data/span-refinement-v1','data/span-refinement-v3']:
        for name in ['validation','test']:prior.extend(load(Path(folder)/(name+'.jsonl')))
    assert_disjoint({'train':splits['train'],'historical_eval':prior})
    output.mkdir()
    for name,rows in splits.items():write_jsonl(output/(name+'.jsonl'),rows)
    write_json(output/'manifest.json',dict(seed=20261031,added=len(added),counts={k:len(v) for k,v in splits.items()},
        fingerprint=digest(json.dumps(splits,sort_keys=True,ensure_ascii=False)),
        policy='V4 validation reused for development selection. V4 test not consulted to design this candidate. Context-only paraphrases, fresh synthetic values, same train families; no holdout families moved into training.'))
if __name__=='__main__':main()
