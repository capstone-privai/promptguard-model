"""Timing utilities shared by CLI/tests. Full pipeline timing excludes model load."""
import math
import os
import platform
import statistics
import time
import torch
import psutil
from privai_model.data.input_builder import encode_candidate
from privai_model.inference.predict import predict_examples
from privai_model.modeling.model import synchronize


def device_status():
    info={'system':platform.system(),'machine':platform.machine(),'torch':torch.__version__,
          'cpu_ok':bool((torch.ones(2,device='cpu')+1).sum().item()==4),
          'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available(),
          'mps_initialized':False}
    if info['mps_available']:
        try:
            x=torch.ones(2,device='mps'); y=(x+1).sum();torch.mps.synchronize()
            info['mps_initialized']=bool(y.cpu().item()==4)
        except Exception as exc: info['mps_initialization_error']=type(exc).__name__
    return info


def rss_snapshot():
    try: return psutil.Process(os.getpid()).memory_info().rss
    except (psutil.Error,OSError): return None


def benchmark(bundle,examples,windows=(64,256,512,1024),batch_sizes=(1,8),warmup=3,repeats=20,require_full_windows=False):
    if repeats<1 or warmup<0 or not examples or not batch_sizes or min(batch_sizes)<1:
        raise ValueError('invalid benchmark inputs')
    if len(examples)<max(batch_sizes):
        raise ValueError('not enough candidates for requested batch size')
    report={'environment':device_status(),'device':str(bundle.device),'model_version':bundle.model_version,
            'scope':'tokenization + feature extraction + transfer + forward + prediction assembly; model load excluded',
            'memory_scope':'process RSS snapshots, not peak; unavailable = null',
            'warmup':warmup,'repeats':repeats,'torch_num_threads':torch.get_num_threads(),'results':[]}
    original=bundle.window_size
    try:
        for window in windows:
            limit=getattr(bundle.model.config,'max_position_embeddings',None)
            if limit and window>limit: raise ValueError('window exceeds backbone maximum')
            bundle.window_size=window
            for size in batch_sizes:
                chunk=examples[:size]
                lengths=[encode_candidate(e,bundle.tokenizer,window)[1]['tokens'] for e in chunk]
                if require_full_windows and any(length!=window for length in lengths):
                    raise ValueError('synthetic context does not fill requested window')
                for _ in range(warmup): predict_examples(bundle,chunk,size)
                before=rss_snapshot();elapsed=[]
                for _ in range(repeats):
                    synchronize(bundle.device); start=time.perf_counter()
                    predict_examples(bundle,chunk,size)
                    synchronize(bundle.device);elapsed.append((time.perf_counter()-start)*1000)
                ordered=sorted(elapsed)
                report['results'].append({'window_size':window,'batch_size':size,'token_lengths':lengths,
                    'mean_latency_ms':statistics.mean(elapsed),'median_latency_ms':statistics.median(elapsed),
                    'p95_latency_ms':ordered[math.ceil(.95*repeats)-1],
                    'throughput_candidates_per_second':size*1000/statistics.mean(elapsed),
                    'rss_before_bytes':before,'rss_after_bytes':rss_snapshot()})
    finally: bundle.window_size=original
    return report
