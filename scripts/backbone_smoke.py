"""One real backbone per process, plumbing only. No ranking or threshold selection."""
import argparse
import gc
import json
from pathlib import Path
import torch
import yaml
from transformers import AutoTokenizer
from privai_model.data.adapter import adapt_records
from privai_model.data.input_builder import encode_candidate
from privai_model.data.synthetic import hard_cases
from privai_model.modeling.model import create_model, resolve_device, seed_everything
from privai_model.modeling.features import extract_features
from privai_model.inference.predict import ModelBundle, predict_candidates, load_model
from privai_model.io import write_json


def run(config, output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('smoke output must be new')
    output.mkdir(parents=True,exist_ok=True)
    stages=('tokenizer','load','tokenize','forward','backward','optimizer_step','save','reload','inference')
    report={'purpose':'synthetic compatibility only; no performance/threshold conclusions',
            'model_name':config['model_name'],'character_features':config.get('character_features',False),
            'stages':{s:'not_run' for s in stages}}
    stage='tokenizer'
    try:
        seed_everything(config.get('seed',42))
        device=resolve_device(config.get('device','cpu'))
        report['device']=str(device)
        kwargs={'local_files_only':config.get('local_files_only',False)}
        if config.get('revision') and config.get('revision_model_name',config['model_name'])==config['model_name']:
            kwargs['revision']=config['revision']
        AutoTokenizer.from_pretrained(config['model_name'],**kwargs)
        report['stages'][stage]='passed'
        stage='load'
        tokenizer,model=create_model(config)
        model.to(device)
        report.update(resolved_revision=getattr(model.config,'_commit_hash',None),
                      architecture=type(model).__name__, parameters=sum(p.numel() for p in model.parameters()))
        report['stages'][stage]='passed'
        stage='tokenize'
        records=hard_cases()
        examples=adapt_records(records[:4])
        encoded=[encode_candidate(e,tokenizer,config['window_size'])[0] for e in examples]
        batch=tokenizer.pad(encoded,padding=True,return_tensors='pt').to(device)
        spec=getattr(model,'feature_spec',None)
        if spec:
            batch['character_features']=torch.tensor([extract_features(e.value,spec) for e in examples],device=device)
        report['feature_dimension']=spec.dimension if spec else 0
        report['stages'][stage]='passed'
        stage='forward'
        model.train()
        labels=torch.tensor([e.label for e in examples],device=device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
        logits=model(**batch).logits
        loss=torch.nn.functional.cross_entropy(logits,labels)
        assert torch.isfinite(loss)
        report['stages'][stage]='passed'
        stage='backward'
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        report['stages'][stage]='passed'
        stage='optimizer_step'
        watched=next(p for n,p in model.named_parameters() if 'classifier' in n and p.grad is not None)
        before=watched.detach().clone()
        optimizer.step()
        assert not torch.equal(before,watched)
        report['stages'][stage]='passed'
        stage='save'
        checkpoint=output/'model'
        model.save_pretrained(checkpoint,safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        metadata={'schema_version':'model-bundle-v1','model_version':'SYNTHETIC-COMPATIBILITY-ONLY',
                  'window_size':config['window_size'],'thresholds':{'0.99':0.5},
                  'deployment_recall_target':0.99,'threshold_status':'synthetic_placeholder_not_selected',
                  'character_features':bool(spec),'feature_dimension':spec.dimension if spec else 0,
                  'feature_spec':spec.to_dict() if spec else None}
        write_json(checkpoint/'bundle.json',metadata)
        original=predict_candidates(ModelBundle(tokenizer,model,device,config['window_size'],metadata['model_version'],.5),records[:10],8)
        report['stages'][stage]='passed'
        # Do not hold optimizer state and two large encoders at once.
        del optimizer,model,logits,loss,watched,before,batch
        gc.collect()
        stage='reload'
        bundle=load_model(checkpoint,device=str(device))
        report['stages'][stage]='passed'
        stage='inference'
        unlabeled=[{k:v for k,v in r.items() if k!='label'} for r in reversed(records[:10])]
        predictions=predict_candidates(bundle,unlabeled,batch_size=8)
        a={r['candidate_id']:r['p_mask'] for r in original}
        b={r['candidate_id']:r['p_mask'] for r in predictions}
        assert set(a)==set(b) and len(predictions)==10
        max_delta=max(abs(a[cid]-b[cid]) for cid in a)
        assert max_delta < 1e-4, 'save/reload or order invariance mismatch'
        report['max_score_delta_after_reload_and_reorder']=max_delta
        report['prediction_count']=len(predictions)
        report['stages'][stage]='passed'
        report['status']='passed'
    except Exception as exc:
        # Only synthetic inputs used; still avoid copying third-party exception text/raw values.
        report.update(status='failed',failed_stage=stage,error_type=type(exc).__name__)
        report['stages'][stage]='failed'
        write_json(output/'smoke.json',report)
        raise
    write_json(output/'smoke.json',report)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--character-features',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    config=yaml.safe_load(Path(a.config).read_text())
    config['device']=a.device
    if a.character_features is not None:
        config['character_features']=a.character_features
    run(config,a.output)
