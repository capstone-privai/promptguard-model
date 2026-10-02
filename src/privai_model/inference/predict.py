from dataclasses import dataclass
from pathlib import Path
import json
import math
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from privai_model.data.adapter import adapt_records, ModelExample
from privai_model.data.input_builder import encode_candidate
from privai_model.evaluation.evaluator import validate_prediction_v1
from privai_model.modeling.model import resolve_device, synchronize, load_checkpoint
from privai_model.modeling.features import extract_features


@dataclass(repr=False)
class ModelBundle:
    tokenizer: object
    model: object
    device: object
    window_size: int
    model_version: str
    threshold: float


def load_model(path, device='auto', recall_target=None):
    path = Path(path)
    metadata = json.loads((path/'bundle.json').read_text())
    target = str(recall_target if recall_target is not None else metadata['deployment_recall_target'])
    threshold = metadata['thresholds'][target]
    dev = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = load_checkpoint(path)
    saved_spec = metadata.get('feature_spec')
    actual_spec = getattr(model, 'feature_spec', None)
    if saved_spec is not None and (actual_spec is None or saved_spec != actual_spec.to_dict()):
        raise ValueError('bundle and checkpoint feature specifications disagree')
    if 'character_features' in metadata and metadata['character_features'] != (actual_spec is not None):
        raise ValueError('bundle feature flag and checkpoint disagree')
    if 'feature_dimension' in metadata and metadata['feature_dimension'] != (actual_spec.dimension if actual_spec else 0):
        raise ValueError('bundle feature dimension and checkpoint disagree')
    model.to(dev).eval()
    return ModelBundle(tokenizer, model, dev, metadata['window_size'], metadata['model_version'], threshold)


def predict_examples(bundle, examples, batch_size=8, threshold=None):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if len({e.candidate_id for e in examples}) != len(examples):
        raise ValueError('duplicate candidate_id')
    threshold = bundle.threshold if threshold is None else threshold
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('invalid threshold')
    rows = []
    bundle.model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start+batch_size]
            synchronize(bundle.device)
            begin = time.perf_counter()
            encoded = [encode_candidate(e, bundle.tokenizer, bundle.window_size)[0] for e in chunk]
            batch = bundle.tokenizer.pad(encoded, padding=True, return_tensors='pt').to(bundle.device)
            spec = getattr(bundle.model, 'feature_spec', None)
            if spec is not None:
                batch['character_features'] = torch.tensor([extract_features(e.value, spec) for e in chunk],
                                                           dtype=torch.float32, device=bundle.device)
            scores = bundle.model(**batch).logits.softmax(-1)[:, 1].cpu().tolist()
            synchronize(bundle.device)
            latency = (time.perf_counter()-begin)*1000/len(chunk)
            for ex, score in zip(chunk, scores):
                rows.append({'schema_version': 'prediction-v1', 'candidate_id': ex.candidate_id,
                             'model_version': bundle.model_version, 'p_mask': score, 'threshold': threshold,
                             'action': 'MASK' if score >= threshold else 'KEEP', 'latency_ms': latency})
    validate_prediction_v1(rows, [e.candidate_id for e in examples])
    return rows


def predict_candidates(model_bundle, candidate_records, batch_size=8):
    """Value-only API: non-value candidates fail explicitly rather than silently losing IDs."""
    records = list(candidate_records)
    if any(not isinstance(r, dict) or r.get('granularity') != 'value' for r in records):
        raise ValueError('inference accepts value granularity only; prefilter non-value candidates')
    examples = adapt_records(records, require_label=False)
    return predict_examples(model_bundle, examples, batch_size)
