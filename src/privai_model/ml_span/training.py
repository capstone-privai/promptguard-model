"""Validation-only epoch/threshold selection, sealed test evaluated once after selection."""
import gc
import json
import time
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig,AutoTokenizer,AutoModelForTokenClassification
from privai_model.modeling.model import seed_everything,resolve_device
from privai_model.io import write_json,write_jsonl
from .schema import load
from .splitter import assert_disjoint
from .bio import windows,align,decode,LABELS
from .runtime import MLSpanDetector
from .evaluation import evaluate,predict_rows,measures
from .rules import RuleSpanDetector


class LengthGroupedBatches:
    """Shuffle equal-length examples and batch order without padding short rows to long rows."""
    def __init__(self,rows,batch_size,seed):
        self.lengths=[len(r['input_ids']) for r in rows]
        self.batch_size=batch_size
        self.generator=torch.Generator().manual_seed(seed)

    def __len__(self):
        return (len(self.lengths)+self.batch_size-1)//self.batch_size

    def __iter__(self):
        indices=torch.randperm(len(self.lengths),generator=self.generator).tolist()
        indices.sort(key=lambda i:self.lengths[i])
        batches=[indices[i:i+self.batch_size] for i in range(0,len(indices),self.batch_size)]
        for i in torch.randperm(len(batches),generator=self.generator).tolist():yield batches[i]


def create_model(config):
    kwargs=dict(local_files_only=config.get('local_files_only',True))
    if config.get('revision'): kwargs['revision']=config['revision']
    tokenizer=AutoTokenizer.from_pretrained(config['model_name'],**kwargs)
    if not tokenizer.is_fast: raise ValueError('fast tokenizer required')
    cfg=AutoConfig.from_pretrained(config['model_name'],**kwargs)
    cfg.num_labels=3;cfg.id2label=LABELS;cfg.label2id={v:k for k,v in LABELS.items()};cfg.reference_compile=False
    model=AutoModelForTokenClassification.from_pretrained(config['model_name'],config=cfg,attn_implementation='eager',**kwargs)
    return tokenizer,model


def encoded_rows(rows,tokenizer,max_length,stride):
    encoded=[];representable=total=0
    for row in rows:
        recovered=set();gold={(s['start'],s['end']) for s in row['spans']}
        for window in windows(tokenizer,row['text'],max_length,stride):
            labels=align(row['text'],window['offsets'],row['spans'],window['special'])
            encoded.append(dict(**window['inputs'],labels=labels))
            recovered.update((s['start'],s['end']) for s in decode(row['text'],window['offsets'],labels))
        representable+=len(gold&recovered);total+=len(gold)
    return encoded,dict(gold_spans=total,exactly_representable_in_a_window=representable,
        boundary_or_window_unrepresentable=total-representable)


def token_metrics(detector,rows):
    gold=set();pred=set()
    from .bio import windows
    with torch.inference_mode():
        for row in rows:
            merged={}
            for w in windows(detector.tokenizer,row['text'],detector.max_length,detector.stride):
                batch={k:torch.tensor([v],device=detector.device) for k,v in w['inputs'].items()}
                probs=detector.model(**batch).logits.softmax(-1)[0].cpu().tolist()
                labels=align(row['text'],w['offsets'],row['spans'],w['special'])
                for (a,b),p,y in zip(w['offsets'],probs,labels):
                    if y==-100: continue
                    key=(row['id'],a,b)
                    if y in (1,2): gold.add(key)
                    old,n=merged.get(key,([0.,0.,0.],0))
                    merged[key]=([x+z for x,z in zip(old,p)],n+1)
            for key,(p,n) in merged.items():
                if max(range(3),key=lambda i:p[i])!=0: pred.add(key)
    return dict(measures(gold,pred),policy='binary secret token, deduplicated absolute offsets, argmax before span threshold')


def train(config):
    if config['epochs']<1 or config['batch_size']<1: raise ValueError('positive epochs/batch size required')
    output=Path(config['output_dir'])
    if output.exists() and any(output.iterdir()): raise FileExistsError('choose a new output directory')
    splits={name:load(Path(config['data_dir'])/(name+'.jsonl')) for name in ('train','validation','test')}
    assert_disjoint(splits)
    for name in ('train','validation'):
        if {bool(r['spans']) for r in splits[name]}!={True,False}: raise ValueError('both classes required')
    output.mkdir(parents=True);write_json(output/'config.json',config)
    seed_everything(config['seed']);torch.set_num_threads(config.get('threads',2))
    device=resolve_device(config.get('device','cpu'))
    tokenizer,model=create_model(config);model.to(device)
    training,alignment=encoded_rows(splits['train'],tokenizer,config['max_length'],config['stride'])
    _,val_alignment=encoded_rows(splits['validation'],tokenizer,config['max_length'],config['stride'])
    write_json(output/'alignment.json',{'train':alignment,'validation':val_alignment})
    def collate(rows):
        batch=tokenizer.pad([{k:v for k,v in r.items() if k!='labels'} for r in rows],return_tensors='pt')
        batch['labels']=torch.tensor([r['labels']+[-100]*(batch['input_ids'].shape[1]-len(r['labels'])) for r in rows])
        return batch
    if config.get('length_grouped_batches',False):
        loader=DataLoader(training,batch_sampler=LengthGroupedBatches(training,config['batch_size'],config['seed']),collate_fn=collate)
    else:
        loader=DataLoader(training,batch_size=config['batch_size'],shuffle=True,collate_fn=collate,
            generator=torch.Generator().manual_seed(config['seed']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config.get('weight_decay',.01))
    weights=config.get('class_weights')
    if weights is not None and (len(weights)!=3 or any(not 0<float(w)<float('inf') for w in weights)):
        raise ValueError('three positive finite class weights required')
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device,dtype=torch.float32) if weights else None,ignore_index=-100)
    best=-1.;stale=0;history=[];checkpoint=output/'model';selected_threshold=None;best_epoch=None
    empty={r['id']:[] for r in splits['validation']}
    start=time.perf_counter()
    for epoch in range(1,config['epochs']+1):
        model.train();loss_sum=0
        for batch_index,batch in enumerate(loader,1):
            batch={k:v.to(device) for k,v in batch.items()};labels=batch.pop('labels')
            optimizer.zero_grad(set_to_none=True)
            logits=model(**batch).logits
            loss=loss_fn(logits.reshape(-1,3),labels.reshape(-1))
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            loss_sum+=loss.item()
            interval=config.get('progress_every_batches',100)
            if interval>0 and batch_index%interval==0:
                print(json.dumps(dict(event='training_progress',epoch=epoch,batch=batch_index,
                    batches=len(loader),mean_loss=loss_sum/batch_index,
                    elapsed_seconds=round(time.perf_counter()-start,1))),flush=True)
        adapter=MLSpanDetector(tokenizer=tokenizer,model=model,device=str(device),max_length=config['max_length'],stride=config['stride'])
        predictions,_=predict_rows(adapter,splits['validation'])
        choices=[]
        for t in config.get('thresholds',[.3,.5,.7,.9]):
            metric=evaluate(splits['validation'],empty,predictions,t)['ml_only']
            choices.append((metric['f2'] or 0.,metric['recall'] or 0.,t,metric))
        f2,_,threshold,metric=max(choices,key=lambda v:v[:3])
        history.append(dict(epoch=epoch,loss=loss_sum/len(loader),threshold=threshold,validation=metric))
        write_json(output/'history.json',history)
        print(json.dumps(history[-1]),flush=True)
        if f2>best:
            best=f2;stale=0;selected_threshold=threshold;best_epoch=epoch
            model.save_pretrained(checkpoint);tokenizer.save_pretrained(checkpoint)
        else: stale+=1
        if stale>=config.get('patience',2): break
    parameters=sum(p.numel() for p in model.parameters())
    write_json(checkpoint/'span_bundle.json',dict(schema_version='ml-span-bundle-v1',max_length=config['max_length'],
        stride=config['stride'],threshold=selected_threshold,threshold_source='synthetic validation F2',
        backbone=config.get('backbone_id',config['model_name']),revision=config.get('backbone_revision',config.get('revision')),
        label_mapping=LABELS,best_epoch=best_epoch,production_ready=False))
    del adapter,optimizer,model,logits,loss,batch;gc.collect()
    detector=MLSpanDetector(checkpoint,device=str(device))
    rules=RuleSpanDetector()
    rule_all={};rule_stats={}
    # These labels are audit strata only; no test-based training selection.
    for name,rows in splits.items():
        rp,lat=predict_rows(rules,rows,True);rule_all[name]=rp
        status=[]
        for row in rows:
            exact={(s['start'],s['end']) for s in rp[row['id']]}
            covered=[any(p['start']<=s['start'] and s['end']<=p['end'] for p in rp[row['id']]) for s in row['spans']]
            status.append(dict(id=row['id'],gold_count=len(row['spans']),
                exact_hits=sum((s['start'],s['end']) in exact for s in row['spans']),coverage_hits=sum(covered),
                rule_findings=rp[row['id']]))
        write_jsonl(output/(name+'-rule-labels.jsonl'),status)
        rule_stats[name]=dict(records=len(rows),gold=sum(s['gold_count'] for s in status),
            exact_hits=sum(s['exact_hits'] for s in status),coverage_hits=sum(s['coverage_hits'] for s in status),latency=lat)
    # Single frozen synthetic test evaluation. Public page data is never fed here.
    rows=splits['test'];ml,latency=predict_rows(detector,rows)
    report=evaluate(rows,rule_all['test'],ml,selected_threshold)
    report.update(notice='Independent synthetic held-out families; NOT public PromptGuard hold-out performance',
        threshold=selected_threshold,threshold_source='validation',best_epoch=best_epoch,
        rule_metadata=rules.metadata,rule_dataset_stats=rule_stats,ml_latency=latency,parameters=parameters,
        checkpoint_bytes=sum(p.stat().st_size for p in checkpoint.glob('*') if p.is_file()),
        elapsed_seconds=time.perf_counter()-start,device=str(device),token_metrics=token_metrics(detector,rows))
    for field in ('source','language','kind'):
        report['by_'+field]={}
        for value in sorted({r[field] for r in rows}):
            subset=[r for r in rows if r[field]==value];ids={r['id'] for r in subset}
            report['by_'+field][value]=evaluate(subset,{k:v for k,v in rule_all['test'].items() if k in ids},
                {k:v for k,v in ml.items() if k in ids},selected_threshold)
    write_json(output/'evaluation.json',report)
    write_jsonl(output/'test-predictions.jsonl',[dict(id=k,spans=v) for k,v in ml.items()])
    return report
