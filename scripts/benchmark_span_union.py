"""Sequential Rule, ML and end-to-end candidate-union latency; no input text exported."""
import argparse
import time
import math
import statistics
from pathlib import Path
import torch
from privai_model.ml_span.schema import load
from privai_model.ml_span.runtime import MLSpanDetector,candidate_union
from privai_model.ml_span.rules import RuleSpanDetector
from privai_model.ml_span.evaluation import predict_rows,evaluate
from privai_model.io import write_json

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ['checkpoint','data','output']:p.add_argument('--'+name,required=True)
 a=p.parse_args();torch.set_num_threads(4)
 rows=load(Path(a.data));rules=RuleSpanDetector();model=MLSpanDetector(a.checkpoint)
 rp,rule_latency=predict_rows(rules,rows,True)
 rule_latency['includes']='Rule scan with source dispatch; sequential loaded model; first scan included'
 mp,ml_latency=predict_rows(model,rows)
 times=[]
 for row in rows:
  start=time.perf_counter()
  rr=rules.scan_text(row['text'],row['source']);mm=model.scan_text(row['text'],threshold=0.)
  union=candidate_union(row['text'],rr,mm,model.threshold)
  times.append((time.perf_counter()-start)*1000)
  assert union==candidate_union(row['text'],rp[row['id']],mp[row['id']],model.threshold)
 ordered=sorted(times)
 result=dict(checkpoint=a.checkpoint,records=len(rows),threads=4,device='cpu',rule_metadata=rules.metadata,
             rule_latency=rule_latency,ml_latency=ml_latency,
             union_latency=dict(unit='ms/input',calls=len(times),mean=statistics.mean(times),
                 p50=statistics.median(times),p95=ordered[math.ceil(.95*len(times))-1],
                 includes='Sequential Rule scan + ML tokenization/all windows/boundary/decoding + candidate union; warm models; excludes model loading and external transport.'),
             evaluation=evaluate(rows,rp,mp,model.threshold),
             repeated_union_predictions_equal=True)
 write_json(a.output,result)
