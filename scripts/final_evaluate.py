"""Evaluate a validation-selected run with its already frozen thresholds."""
import argparse
import json
from pathlib import Path
from privai_model.data.adapter import load_examples,iter_jsonl
from privai_model.data.split import restore_split
from privai_model.inference.predict import load_model,predict_examples
from privai_model.evaluation.evaluator import evaluate
from privai_model.io import write_json,write_jsonl


def finalize(data,run,output,device='auto',batch_size=8):
    run=Path(run);output=Path(output)
    manifest=json.loads((run/'split_manifest.json').read_text())
    splits=restore_split(load_examples(data),manifest)
    metadata=json.loads((run/'model/bundle.json').read_text())
    if metadata.get('dataset_fingerprint')!=manifest['dataset_fingerprint']:
        raise ValueError('model/split fingerprint mismatch')
    if output.exists() and any(output.iterdir()): raise FileExistsError('final evaluation output must be new')
    bundle=load_model(run/'model',device=device)
    thresholds=metadata['thresholds']
    validation=list(iter_jsonl(run/'validation_predictions.jsonl'))
    test=predict_examples(bundle,splits['test'],batch_size)
    output.mkdir(parents=True,exist_ok=True)
    write_jsonl(output/'test_predictions.jsonl',test)
    report={'threshold_source':'frozen model bundle (validation selected)', 'thresholds':thresholds,
            'model_version':metadata['model_version'],'threshold_status':metadata.get('threshold_status','unspecified'),
            'validation':{k:evaluate(splits['validation'],validation,t) for k,t in thresholds.items()},
            'test':{k:evaluate(splits['test'],test,t) for k,t in thresholds.items()}}
    write_json(output/'evaluation.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='auto');p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args();finalize(a.data,a.run,a.output,a.device,a.batch_size)
    print('Final evaluation written using frozen validation thresholds')
