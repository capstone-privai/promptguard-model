"""Compare frozen checkpoints without tuning on any test labels."""
from pathlib import Path
import torch
from privai_model.io import write_json
from privai_model.ml_span.schema import load
from privai_model.ml_span.runtime import MLSpanDetector
from privai_model.ml_span.rules import RuleSpanDetector
from privai_model.ml_span.evaluation import predict_rows, evaluate

def main():
    torch.set_num_threads(4)
    rules=RuleSpanDetector()
    result={}
    for split,path in [('fresh','data/span-refinement-v3/test.jsonl'),('development','data/span-refinement-v1/test.jsonl'),('original_development','data/span-synthetic-v1/test.jsonl')]:
        rows=load(Path(path));rp,rl=predict_rows(rules,rows,True)
        result[split]={}
        for name,checkpoint in [('previous','runs/span-mmbert-refine-v1/model'),('new','runs/span-mmbert-context-v3/model')]:
            detector=MLSpanDetector(checkpoint)
            mp,lat=predict_rows(detector,rows)
            result[split][name]=dict(evaluate(rows,rp,mp,detector.threshold),threshold=detector.threshold,ml_latency=lat,rule_latency=rl)
            del detector
    write_json('reports/ml_span/context-v3-comparison.json',result)
if __name__=='__main__': main()
