"""Frozen checkpoint comparison; validation decision precedes sealed test access.

Reports contain aggregate counts and error coordinates/IDs only, never text or values.
Historical evaluations are explicitly development checks, not independent test results.
"""
import argparse
import gc
from pathlib import Path
import torch
from privai_model.io import write_json
from privai_model.ml_span.schema import load
from privai_model.ml_span.runtime import MLSpanDetector
from privai_model.ml_span.rules import RuleSpanDetector
from privai_model.ml_span.evaluation import evaluate, predict_rows


def errors(rows,predictions,threshold):
    result=[]
    for row in rows:
        gold={(s['start'],s['end']) for s in row['spans']}
        pred={(s['start'],s['end']) for s in predictions[row['id']] if s['score']>=threshold}
        if gold==pred: continue
        result.append(dict(id=row['id'],family=row['family_id'],source=row['source'],language=row['language'],
            kind=row['kind'],value_format=row.get('value_format','legacy'),
            missing=[dict(start=a,end=b,overlapping_prediction=any(max(a,c)<min(b,d) for c,d in pred)) for a,b in sorted(gold-pred)],
            false_positive=[dict(start=a,end=b,overlapping_gold=any(max(a,c)<min(b,d) for c,d in gold)) for a,b in sorted(pred-gold)]))
    return result


def comparison(rows,rule_predictions,checkpoints):
    report={}
    for name,path in checkpoints.items():
        detector=MLSpanDetector(path)
        predictions,latency=predict_rows(detector,rows)
        threshold=detector.threshold
        result=evaluate(rows,rule_predictions,predictions,threshold)
        result.update(threshold=threshold,checkpoint=str(path),ml_latency=latency,errors=errors(rows,predictions,threshold))
        result['error_summary']=dict(
            missed_with_no_overlap=sum(not s['overlapping_prediction'] for r in result['errors'] for s in r['missing']),
            missed_with_inexact_overlap=sum(s['overlapping_prediction'] for r in result['errors'] for s in r['missing']),
            false_positive_no_gold_overlap=sum(not s['overlapping_gold'] for r in result['errors'] for s in r['false_positive']),
            false_positive_overlapping_gold=sum(s['overlapping_gold'] for r in result['errors'] for s in r['false_positive']),
            negative_rows_with_findings=sum(r['kind']=='hard_negative' and bool(r['false_positive']) for r in result['errors']))
        for field in ['family_id','source','language','value_format','kind']:
            result['by_'+field]={}
            for value in sorted({r.get(field,'legacy') for r in rows}):
                subset=[r for r in rows if r.get(field,'legacy')==value];ids={r['id'] for r in subset}
                result['by_'+field][value]=evaluate(subset,{k:v for k,v in rule_predictions.items() if k in ids},
                    {k:v for k,v in predictions.items() if k in ids},threshold)
        report[name]=result
        del detector;gc.collect()
    return report


def run(previous,candidate,data_dir,output,unrefined=None,alternative=None,development=None,additional=None):
    torch.set_num_threads(4)
    checkpoints={'previous':previous,'candidate':candidate}
    if unrefined is not None: checkpoints['without_boundary']=unrefined
    if alternative is not None: checkpoints['intermediate_v4']=alternative
    for entry in additional or []:
        name,path=entry.split('=',1)
        if not name or name in checkpoints:raise ValueError('unique checkpoint name required')
        checkpoints[name]=path
    rules=RuleSpanDetector()
    report=dict(notice='Synthetic only. Validation decision frozen before test comparison. All thresholds loaded from saved checkpoints.',rule_metadata=rules.metadata,datasets={})
    validation=load(Path(data_dir)/'validation.jsonl')
    rp,lat=predict_rows(rules,validation,True)
    result=comparison(validation,rp,checkpoints)
    report['datasets']['validation']=result
    old=result['previous']['ml_only']
    eligible={name:(r['ml_only']['f2'] or 0)>=(old['f2'] or 0) and (r['ml_only']['precision'] or 0)>=(old['precision'] or 0)
        for name,r in result.items()}
    selected=max((name for name in result if eligible[name]),
        key=lambda name:(result[name]['ml_only']['f2'] or 0,result[name]['ml_only']['precision'] or 0))
    report['validation_decision']=dict(eligible=eligible,policy='Validation F2 and precision must both be no worse than previous checkpoint; select greatest validation F2, precision tie-break. No test-based threshold or checkpoint selection.',
        selected=selected,checkpoint=checkpoints[selected])
    write_json(output,report)
    datasets={'fresh_test':str(Path(data_dir)/'test.jsonl'),
        'v3_development':'data/span-refinement-v3/test.jsonl',
        'v1_development':'data/span-refinement-v1/test.jsonl',
        'original_development':'data/span-synthetic-v1/test.jsonl',
        'negative_challenge':'data/span-negative-challenge-v1/test.jsonl'}
    for entry in development or []:
        name,path=entry.split('=',1)
        if not name.endswith('_development') or name in datasets:
            raise ValueError('unique *_development dataset name required')
        datasets[name]=path
    for name,path in datasets.items():
        rows=load(Path(path));rp,lat=predict_rows(rules,rows,True)
        report['datasets'][name]=comparison(rows,rp,checkpoints)
        report.setdefault('rule_latency',{})[name]=lat
        write_json(output,report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--unrefined')
    p.add_argument('--alternative')
    p.add_argument('--development',action='append',default=[],metavar='NAME_development=PATH')
    p.add_argument('--additional',action='append',default=[],metavar='NAME=CHECKPOINT')
    a=p.parse_args();run(a.previous,a.candidate,a.data_dir,a.output,a.unrefined,a.alternative,a.development,a.additional)
