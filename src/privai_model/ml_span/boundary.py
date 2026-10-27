"""Optional learned character-boundary trimming, independent of secret detection.

The refiner cannot create candidates, expand spans, or change their confidence.
It learns only from synthetic training annotations; acceptance is validation-gated.
"""
import unicodedata
import torch
from torch import nn

RADIUS=12
MAX_TRIM=8
CATEGORIES=['Lu','Ll','Lt','Lm','Lo','Mn','Mc','Me','Nd','Nl','No','Pc','Pd','Ps','Pe','Pi','Pf','Po','Sm','Sc','Sk','So','Zs','Zl','Zp','Cc','Cf','Cs','Co','Cn']


def encode(text,position,direction):
    chars=[];categories=[]
    for i in range(position-RADIUS,position+RADIUS):
        if 0<=i<len(text):
            chars.append(1+ord(text[i])%2048)
            categories.append(1+CATEGORIES.index(unicodedata.category(text[i])))
        else: chars.append(0);categories.append(0)
    return chars,categories,direction


class BoundaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.characters=nn.Embedding(2049,12,padding_idx=0)
        self.categories=nn.Embedding(31,4,padding_idx=0)
        self.direction=nn.Embedding(2,8)
        self.layers=nn.Sequential(nn.Linear(RADIUS*2*16+8,96),nn.GELU(),nn.Linear(96,MAX_TRIM+1))

    def forward(self,chars,categories,direction):
        encoded=torch.cat([self.characters(chars),self.categories(categories)],dim=-1).flatten(1)
        return self.layers(torch.cat([encoded,self.direction(direction)],dim=-1))


def tensors(encoded,device='cpu'):
    return tuple(torch.tensor(v,dtype=torch.long,device=device) for v in zip(*encoded))


class BoundaryRefiner:
    def __init__(self,path,threshold=.95,device='cpu'):
        self.model=BoundaryModel().to(device)
        self.model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
        self.model.eval();self.threshold=threshold;self.device=device

    def refine(self,text,spans):
        if not spans:return []
        inputs=[encode(text,s['start'] if direction==0 else s['end'],direction) for s in spans for direction in (0,1)]
        with torch.inference_mode(): probs=self.model(*tensors(inputs,self.device)).softmax(-1).cpu()
        result=[]
        for i,span in enumerate(spans):
            shifts=[]
            for row in probs[2*i:2*i+2]:
                confidence,shift=row.max(0)
                shifts.append(int(shift) if float(confidence)>=self.threshold else 0)
            a,b=span['start']+shifts[0],span['end']-shifts[1]
            if a>=b:a,b=span['start'],span['end']
            result.append(dict(span,start=a,end=b))
        return result
