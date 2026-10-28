"""Train a small optional character head on gold token-expanded synthetic boundaries."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from privai_model.io import write_json
from privai_model.ml_span.boundary import BoundaryModel,encode,tensors,MAX_TRIM
from privai_model.ml_span.bio import windows
from privai_model.ml_span.schema import load


def samples(rows,tokenizer):
    xs=[];ys=[];unsupported=0
    for row in rows:
        text=row['text']
        offsets={offset for w in windows(tokenizer,text) for offset in w['offsets'] if offset[0]!=offset[1]}
        for span in row['spans']:
            start,end=span['start'],span['end']
            left=min(a for a,b in offsets if a<=start<b)
            right=max(b for a,b in offsets if a<end<=b)
            while left<start and text[left].isspace():left+=1
            while right>end and text[right-1].isspace():right-=1
            for position,shift,direction in [(left,start-left,0),(right,right-end,1)]:
                if not 0<=shift<=MAX_TRIM:unsupported+=1;continue
                xs.append(encode(text,position,direction));ys.append(shift)
    return tensors(xs),torch.tensor(ys),unsupported


def train(output='runs/span-boundary-v1',data_dir='data/span-context-v4',tokenizer='runs/span-mmbert-context-v3/model'):
    root=Path(output)
    if root.exists():raise FileExistsError(root)
    root.mkdir();torch.set_num_threads(1);torch.manual_seed(872)
    tok=AutoTokenizer.from_pretrained(tokenizer,local_files_only=True)
    data=Path(data_dir)
    tx,ty,tu=samples(load(data/'train.jsonl'),tok)
    vx,vy,vu=samples(load(data/'validation.jsonl'),tok)
    model=BoundaryModel();optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    counts=torch.bincount(ty,minlength=MAX_TRIM+1).float()
    weights=(counts.clamp_min(1).sum()/counts.clamp_min(1)).sqrt().clamp(max=10)
    loss_fn=torch.nn.CrossEntropyLoss(weight=weights)
    best=-1.;history=[]
    for epoch in range(1,41):
        model.train();order=torch.randperm(len(ty));total=0.
        for idx in order.split(64):
            optimizer.zero_grad(set_to_none=True);loss=loss_fn(model(*(x[idx] for x in tx)),ty[idx])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();total+=float(loss.detach())
        model.eval()
        with torch.inference_mode():
            probs=model(*vx).softmax(-1)
            choices=[]
            for threshold in [.8,.9,.95,.98,.99]:
                conf,pred=probs.max(-1);pred=pred.masked_fill(conf<threshold,0)
                correct=int((pred==vy).sum());wrong_trim=int(((pred!=0)&(pred!=vy)).sum())
                # Penalize introducing wrong trimming more than failing to trim.
                choices.append((correct-3*wrong_trim,threshold,correct,wrong_trim))
            score,threshold,correct,wrong_trim=max(choices)
        history.append(dict(epoch=epoch,loss=total,validation_correct=correct,validation_total=len(vy),wrong_trim=wrong_trim,threshold=threshold,score=score))
        if score>best:
            best=score;torch.save(model.state_dict(),root/'boundary.pt')
            write_json(root/'selection.json',dict(history[-1],train_samples=len(ty),train_unrepresentable=tu,validation_unrepresentable=vu,policy='Validation character boundary score only; full detector validation acceptance still required'))
        write_json(root/'history.json',history)
    print(json.dumps(json.loads((root/'selection.json').read_text())))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='runs/span-boundary-v1')
    p.add_argument('--data-dir',default='data/span-context-v4')
    p.add_argument('--tokenizer',default='runs/span-mmbert-context-v3/model')
    a=p.parse_args();train(a.output,a.data_dir,a.tokenizer)
