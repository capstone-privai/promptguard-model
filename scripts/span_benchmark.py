"""Measure sequential Rule / ML / Rule+ML+Union paths on independent synthetic inputs."""
import argparse
import math
import statistics
import time
from pathlib import Path
import torch
from privai_model.ml_span.runtime import MLSpanDetector,candidate_union
from privai_model.ml_span.rules import RuleSpanDetector
from privai_model.io import write_json


def benchmark(checkpoint,output,repeat=5):
    torch.set_num_threads(4)
    start=time.perf_counter();ml=MLSpanDetector(checkpoint);ml_init=(time.perf_counter()-start)*1000
    start=time.perf_counter();rules=RuleSpanDetector();rule_init=(time.perf_counter()-start)*1000
    report=dict(checkpoint=str(checkpoint),device='cpu',threads=4,repeat=repeat,
        ml_initialization_ms=ml_init,rule_initialization_ms=rule_init,inputs={})
    for name,text in {'short':'Synthetic benchmark. Authenticate using "BenchOnly!8wZ5pQ".',
        'long':'Synthetic benchmark status\n'*600+'Authenticate using "BenchOnly!3pZ8wQ".'}.items():
        for _ in range(2):
            ml.scan_text(text);rules.scan_text(text)
        methods={
            'rule':lambda:rules.scan_text(text),
            'ml':lambda:ml.scan_text(text),
            'sequential_union':lambda:candidate_union(text,rules.scan_text(text),ml.scan_text(text),ml.threshold),
        }
        report['inputs'][name]=dict(characters=len(text),methods={})
        for method,fn in methods.items():
            times=[]
            for _ in range(repeat):
                start=time.perf_counter();fn();times.append((time.perf_counter()-start)*1000)
            report['inputs'][name]['methods'][method]=dict(mean_ms=statistics.mean(times),p50_ms=statistics.median(times),
                p95_ms=sorted(times)[min(len(times)-1,math.ceil(.95*len(times))-1)])
    write_json(output,report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--repeat',type=int,default=5);a=p.parse_args();benchmark(a.checkpoint,a.output,a.repeat)
