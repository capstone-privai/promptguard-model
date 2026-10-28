"""Validation-only threshold selection constrained by a frozen reference model."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from privai_model.io import write_json
from privai_model.ml_span.schema import load
from privai_model.ml_span.runtime import MLSpanDetector
from privai_model.ml_span.evaluation import predict_rows, evaluate


def choose_threshold(rows,reference,candidate,reference_threshold,thresholds):
    empty={r['id']:[] for r in rows}
    baseline=evaluate(rows,empty,reference,reference_threshold)['ml_only']
    choices=[]
    for threshold in thresholds:
        metric=evaluate(rows,empty,candidate,threshold)['ml_only']
        eligible=(metric['precision'] is not None and baseline['precision'] is not None
                  and metric['precision']>=baseline['precision'] and metric['recall']>=baseline['recall']
                  and metric['f2']>=baseline['f2'])
        choices.append(dict(threshold=threshold,metric=metric,eligible=eligible))
    eligible=[c for c in choices if c['eligible']]
    selected=max(eligible,key=lambda c:(c['metric']['f2'],c['metric']['precision'],c['threshold'])) if eligible else None
    return dict(reference=baseline,choices=choices,selected=selected,
                policy='Validation only: precision, recall and F2 must not decrease versus reference; maximize F2, precision, threshold.')


def main(a):
    torch.set_num_threads(4)
    output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    rows=load(Path(a.data_dir)/'validation.jsonl')
    reference=MLSpanDetector(a.reference);rp,_=predict_rows(reference,rows)
    candidate=MLSpanDetector(a.checkpoint);cp,_=predict_rows(candidate,rows)
    thresholds=sorted({s['score'] for spans in cp.values() for s in spans}) if a.observed_thresholds else a.thresholds
    result=choose_threshold(rows,rp,cp,reference.threshold,thresholds)
    result['search']='all observed validation confidence cutoffs' if a.observed_thresholds else 'configured grid'
    result.update(validation_only=True,reference_checkpoint=a.reference,candidate_checkpoint=a.checkpoint)
    if result['selected']:
        shutil.copytree(a.checkpoint,output)
        meta=json.loads((output/'span_bundle.json').read_text())
        meta['threshold']=result['selected']['threshold']
        meta['threshold_source']='synthetic validation F2 constrained by reference precision and recall'
        meta['calibration_reference']=a.reference
        write_json(output/'span_bundle.json',meta)
        result['checkpoint']=str(output)
    write_json(a.report,result)
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['reference','checkpoint','data-dir','output','report']:p.add_argument('--'+name,required=True)
    p.add_argument('--thresholds',type=float,nargs='+',default=[.5,.7,.8,.9,.95,.98])
    p.add_argument('--observed-thresholds',action='store_true')
    main(p.parse_args())
