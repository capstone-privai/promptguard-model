"""Publish the optional boundary bundle only after full-detector validation improves."""
import json
import argparse
import shutil
from pathlib import Path
import torch
from privai_model.io import write_json
from privai_model.ml_span.schema import load
from privai_model.ml_span.runtime import MLSpanDetector
from privai_model.ml_span.boundary import BoundaryRefiner
from privai_model.ml_span.evaluation import predict_rows,evaluate


def main(checkpoint='runs/span-mmbert-context-v4/model',data_dir='data/span-context-v4',
         output='runs/span-mmbert-context-v4-boundary/model',report_path='reports/ml_span/context-v4-boundary-validation.json',boundary_dir='runs/span-boundary-v1'):
    torch.set_num_threads(4)
    rows=load(Path(data_dir)/'validation.jsonl')
    root=Path(checkpoint)
    model=MLSpanDetector(root)
    base,lat=predict_rows(model,rows)
    config=json.loads((Path(boundary_dir)/'selection.json').read_text())
    refiner=BoundaryRefiner(Path(boundary_dir)/'boundary.pt',config['threshold'])
    corrected={r['id']:refiner.refine(r['text'],base[r['id']]) for r in rows}
    empty={r['id']:[] for r in rows}
    old=evaluate(rows,empty,base,model.threshold)['ml_only']
    new=evaluate(rows,empty,corrected,model.threshold)['ml_only']
    eligible=new['f2']>old['f2'] and (new['precision'] or 0)>=(old['precision'] or 0)
    report=dict(validation_only=True,baseline=old,with_boundary=new,eligible=eligible,
        character_selection=config,policy='Strictly greater exact validation F2 and no reduction in exact precision; test unopened by this script.')
    if eligible:
        output=Path(output)
        if output.exists():raise FileExistsError(output)
        shutil.copytree(root,output)
        shutil.copyfile(Path(boundary_dir)/'boundary.pt',output/'boundary.pt')
        meta=json.loads((output/'span_bundle.json').read_text())
        meta['boundary_refiner']=dict(schema_version='character-boundary-v1',threshold=config['threshold'])
        write_json(output/'span_bundle.json',meta)
        # Check persisted bundle returns exactly the validation-gated predictions.
        loaded=MLSpanDetector(output)
        actual,_=predict_rows(loaded,rows)
        assert actual==corrected
        report['checkpoint']=str(output)
    write_json(report_path,report)
    print(json.dumps(report))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='runs/span-mmbert-context-v4/model')
    p.add_argument('--data-dir',default='data/span-context-v4')
    p.add_argument('--output',default='runs/span-mmbert-context-v4-boundary/model')
    p.add_argument('--report',default='reports/ml_span/context-v4-boundary-validation.json')
    p.add_argument('--boundary-dir',default='runs/span-boundary-v1')
    a=p.parse_args();main(a.checkpoint,a.data_dir,a.output,a.report,a.boundary_dir)
