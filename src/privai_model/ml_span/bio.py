"""Fast tokenizer offsets; overlapping windows aggregate before global BIO decoding."""
import math
from .schema import validate_spans

LABELS={0:'O',1:'B-SECRET',2:'I-SECRET'}

def windows(tokenizer,text,max_length=512,stride=128):
    if not tokenizer.is_fast: raise ValueError('fast tokenizer required')
    if not 0<=stride<max_length-tokenizer.num_special_tokens_to_add(False):
        raise ValueError('stride must fit content window')
    encoded=tokenizer(text,return_offsets_mapping=True,return_overflowing_tokens=True,
        truncation=True,max_length=max_length,stride=stride,return_special_tokens_mask=True)
    result=[]
    for i,offsets in enumerate(encoded['offset_mapping']):
        offsets=[tuple(x) for x in offsets]
        if any(not 0<=a<=b<=len(text) for a,b in offsets): raise ValueError('token offset out of range')
        result.append(dict(inputs={k:encoded[k][i] for k in ('input_ids','attention_mask')},
            offsets=offsets,special=encoded['special_tokens_mask'][i]))
    return result

def align(text,offsets,spans,special=None):
    validate_spans(text,spans)
    labels=[];previous=None
    for i,(a,b) in enumerate(offsets):
        if a==b or (special and special[i]): labels.append(-100);continue
        hits=[j for j,s in enumerate(spans) if max(a,s['start'])<min(b,s['end'])]
        if len(hits)>1: raise ValueError('token crosses two gold spans; BIO cannot represent annotation')
        current=hits[0] if hits else None
        if current is None: labels.append(0)
        else:
            # At a window's clipped left edge the token is an I, not a false B.
            labels.append(1 if current!=previous and a<=spans[current]['start']<b else 2)
        previous=current
    return labels

def decode(text,offsets,labels,probabilities=None):
    if len(offsets)!=len(labels) or (probabilities is not None and len(labels)!=len(probabilities)):
        raise ValueError('BIO length mismatch')
    output=[];active=None;scores=[]
    def flush():
        nonlocal active,scores
        if active:
            a,b=active
            # Tokenizers may attach boundary whitespace. Never strip punctuation/Korean suffixes.
            while a<b and text[a].isspace(): a+=1
            while b>a and text[b-1].isspace(): b-=1
            if a<b: output.append(dict(detector='ml',secret_type='SECRET',start=a,end=b,score=sum(scores)/len(scores)))
        active=None;scores=[]
    for i,((a,b),label) in enumerate(zip(offsets,labels)):
        if not 0<=a<=b<=len(text) or label not in (-100,0,1,2): raise ValueError('invalid BIO input')
        if a==b or label==-100: continue
        score=probabilities[i] if probabilities is not None else 1.0
        if not math.isfinite(score) or not 0<=score<=1: raise ValueError('invalid probability')
        if label==0: flush();continue
        if label==1 or active is None: flush();active=[a,b]
        else: active[1]=max(active[1],b)
        scores.append(score)
    flush()
    validate_spans(text,output,True)
    return output

def aggregate(text,predictions):
    # Same absolute token coordinates across windows; averaging avoids clipping/duplicate spans.
    totals={}
    for offsets,probs in predictions:
        for (a,b),p in zip(offsets,probs):
            if a==b: continue
            key=(a,b)
            total,n=totals.get(key,([0.,0.,0.],0))
            totals[key]=([x+float(y) for x,y in zip(total,p)],n+1)
    offsets=sorted(totals)
    probs=[[x/totals[k][1] for x in totals[k][0]] for k in offsets]
    labels=[max(range(3),key=lambda j:p[j]) for p in probs]
    # Float32 softmax can sum slightly above one when probabilities saturate.
    # Renormalize accumulated distributions and clamp only the derived score.
    if any(len(p)!=3 or any(not math.isfinite(v) or v<0 for v in p) or not math.isclose(sum(p),1.,abs_tol=1e-5) for p in probs):
        raise ValueError('invalid token probability distribution')
    scores=[min(1.,max(0.,(p[1]+p[2])/sum(p))) for p in probs]
    return decode(text,offsets,labels,scores)
