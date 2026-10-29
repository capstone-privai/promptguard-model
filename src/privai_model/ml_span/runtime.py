"""Independent local inference and union. Never log text or embed values in findings."""
import json
import math
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForTokenClassification
from privai_model.modeling.model import resolve_device
from .bio import windows,aggregate,LABELS
from .schema import validate_spans
from .function_words import POLICY as FUNCTION_WORD_POLICY, filter_function_words


def threshold_check(value):
    if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('threshold must be finite in [0,1]')


def candidate_union(text,rule,ml,threshold):
    """Keep each rule candidate; allow ML alone. Deduplicate exact spans, retain overlaps.

    Mask intervals are a SEPARATE operation so a broad ML span cannot delete rule TPs.
    """
    threshold_check(threshold);validate_spans(text,rule);validate_spans(text,ml,True)
    result={}
    for finding in rule+[s for s in ml if s['score']>=threshold]:
        key=(finding['start'],finding['end'])
        if key not in result:
            result[key]={k:finding[k] for k in ('start','end','score') if k in finding}
            result[key].update(secret_type='SECRET',detectors=[])
        detector=finding.get('detector','rule')
        if detector not in result[key]['detectors']: result[key]['detectors'].append(detector)
    return [result[k] for k in sorted(result)]


def mask_intervals(text,findings):
    validate_spans(text,findings)
    result=[]
    for s in sorted(findings,key=lambda s:(s['start'],s['end'])):
        if result and s['start']<result[-1]['end']: result[-1]['end']=max(result[-1]['end'],s['end'])
        else: result.append(dict(start=s['start'],end=s['end']))
    return result


class MLSpanDetector:
    def __init__(self,checkpoint=None,device='cpu',*,tokenizer=None,model=None,max_length=512,stride=128,threshold=0.5,candidate_filter=None):
        try:
            boundary_config=None
            if checkpoint is not None:
                meta=json.loads((Path(checkpoint)/'span_bundle.json').read_text())
                if meta['schema_version']!='ml-span-bundle-v1': raise ValueError()
                max_length,stride,threshold=meta['max_length'],meta['stride'],meta['threshold']
                boundary_config=meta.get('boundary_refiner')
                candidate_filter=meta.get('candidate_filter')
                tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
                model=AutoModelForTokenClassification.from_pretrained(checkpoint,local_files_only=True,attn_implementation='eager')
            threshold_check(threshold)
            if candidate_filter not in (None, FUNCTION_WORD_POLICY): raise ValueError('unknown candidate filter')
            if tokenizer is None or model is None or not tokenizer.is_fast or model.config.id2label!=LABELS:
                raise ValueError()
            self.device=resolve_device(device)
            self.model=model.to(self.device).eval();self.tokenizer=tokenizer
            self.max_length=max_length;self.stride=stride;self.threshold=threshold
            self.candidate_filter=candidate_filter
            self.boundary_refiner=None
            if boundary_config is not None:
                from .boundary import BoundaryRefiner
                if boundary_config.get('schema_version')!='character-boundary-v1': raise ValueError()
                # Fixed relative filename: bundle metadata cannot select another path.
                threshold_check(boundary_config['threshold'])
                self.boundary_refiner=BoundaryRefiner(Path(checkpoint)/'boundary.pt',
                    threshold=boundary_config['threshold'],device=self.device)
            windows(tokenizer,'',max_length,stride)
        except Exception:
            raise RuntimeError('ML span model initialization failed') from None

    def scan_text(self,text,threshold=None):
        if not isinstance(text,str): raise ValueError('text must be string')
        threshold=self.threshold if threshold is None else threshold
        threshold_check(threshold)
        if not text: return []
        try:
            predictions=[]
            with torch.inference_mode():
                for window in windows(self.tokenizer,text,self.max_length,self.stride):
                    batch={k:torch.tensor([v],device=self.device) for k,v in window['inputs'].items()}
                    probs=self.model(**batch).logits.softmax(-1)[0].cpu().tolist()
                    predictions.append((window['offsets'],probs))
            spans=aggregate(text,predictions)
            if self.boundary_refiner is not None:
                spans=self.boundary_refiner.refine(text,spans)
                validate_spans(text,spans,True)
            spans=[s for s in spans if s['score']>=threshold]
            if self.candidate_filter==FUNCTION_WORD_POLICY:
                spans=filter_function_words(text,spans)
            return spans
        except Exception:
            raise RuntimeError('ML span inference failed; caller must fail closed') from None
