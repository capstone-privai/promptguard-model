import argparse
import json
import torch
from privai_model.benchmarking import benchmark,device_status
from privai_model.data.adapter import load_examples,adapt_records
from privai_model.data.synthetic import hard_cases
from privai_model.inference.predict import load_model
from privai_model.io import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model')
    data=p.add_mutually_exclusive_group()
    data.add_argument('--data')
    data.add_argument('--synthetic-long',action='store_true')
    p.add_argument('--check-devices',action='store_true')
    p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[1,8])
    p.add_argument('--windows',type=int,nargs='+',default=[64,256,512,1024])
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--threads',type=int)
    p.add_argument('--output',default='benchmark.json')
    a=p.parse_args()
    if a.threads: torch.set_num_threads(a.threads)
    if a.check_devices:
        report=device_status()
    else:
        if not a.model or not (a.data or a.synthetic_long): p.error('--model and --data or --synthetic-long required')
        bundle=load_model(a.model,a.device)
        examples=adapt_records(hard_cases(True)) if a.synthetic_long else load_examples(a.data,require_label=False)
        report=benchmark(bundle,examples,a.windows,a.batch_sizes,a.warmup,a.repeats,a.synthetic_long)
    write_json(a.output,report)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
