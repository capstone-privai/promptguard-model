"""Exact span contributions use identity sets; coverage and overlap are separate."""
import time
import math
import statistics
from .schema import validate,validate_spans
from .runtime import candidate_union,mask_intervals


def measures(gold,predicted):
    tp=len(gold&predicted);fp=len(predicted-gold);fn=len(gold-predicted)
    precision=tp/len(predicted) if predicted else None
    recall=tp/len(gold) if gold else None
    return dict(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        f2=5*tp/(5*tp+fp+4*fn) if 5*tp+fp+4*fn else None)


def evaluate(rows,rule_predictions,ml_predictions,threshold=0.5):
    ids={r['id'] for r in rows}
    if not rows or len(ids)!=len(rows) or set(rule_predictions)!=ids or set(ml_predictions)!=ids:
        raise ValueError('nonempty complete unique prediction coverage required')
    gold=set();rules=set();ml=set();unions=set();masked=set();overlap=covered=0
    no_overlap_misses=set();coverage_misses=set()
    def keys(row,spans): return {(row['id'],s['start'],s['end']) for s in spans}
    for row in rows:
        validate(row);rid=row['id'];text=row['text']
        r,m=rule_predictions[rid],ml_predictions[rid]
        validate_spans(text,r);validate_spans(text,m,True)
        m=[s for s in m if s['score']>=threshold]
        u=candidate_union(text,r,m,threshold)
        gold.update(keys(row,row['spans']));rules.update(keys(row,r));ml.update(keys(row,m));unions.update(keys(row,u))
        for s in row['spans']:
            key=(rid,s['start'],s['end'])
            if not any(max(s['start'],p['start'])<min(s['end'],p['end']) for p in r): no_overlap_misses.add(key)
            if not any(p['start']<=s['start'] and s['end']<=p['end'] for p in r): coverage_misses.add(key)
        masked.update(keys(row,mask_intervals(text,u)))
        overlap+=sum(any(max(s['start'],p['start'])<min(s['end'],p['end']) for p in m) for s in row['spans'])
        covered+=sum(any(p['start']<=s['start'] and p['end']>=s['end'] for p in u) for s in row['spans'])
    missed=gold-rules;added=(gold&ml)-rules
    return dict(records=len(rows),gold_spans=len(gold),rule_only=measures(gold,rules),ml_only=measures(gold,ml),
        candidate_union=measures(gold,unions),merged_mask_intervals=measures(gold,masked),
        rule_miss_count=len(missed),rule_miss_recall=len(added)/len(missed) if missed else None,
        rule_no_overlap_miss_count=len(no_overlap_misses),
        rule_no_overlap_miss_recall=len(no_overlap_misses&ml)/len(no_overlap_misses) if no_overlap_misses else None,
        rule_coverage_miss_count=len(coverage_misses),
        rule_coverage_miss_recall=len(coverage_misses&ml)/len(coverage_misses) if coverage_misses else None,
        ml_added_true_positives=len(added),ml_induced_false_positives=len((ml-gold)-rules),
        ml_overlap_recall=overlap/len(gold) if gold else None,combined_coverage_recall=covered/len(gold) if gold else None)


def predict_rows(detector,rows,rule=False):
    predictions={};times=[]
    for row in rows:
        begin=time.perf_counter()
        predictions[row['id']]=detector.scan_text(row['text'],row['source']) if rule else detector.scan_text(row['text'],threshold=0.)
        times.append((time.perf_counter()-begin)*1000)
    ordered=sorted(times)
    return predictions,dict(unit='ms/input',calls=len(times),mean=statistics.mean(times) if times else None,
        p50=statistics.median(times) if times else None,p95=ordered[min(len(times)-1,math.ceil(.95*len(times))-1)] if times else None,
        includes='tokenization, all windows, decoding; sequential warm model; first scan included')
