"""Greedy generation -> exact spans with per-item confidence. Callers must fail closed
on `error` (malformed or truncated answers return no spans, never a guess)."""
import time
import torch
from transformers import GenerationConfig
from . import format as fmt
from .scoring import token_char_ranges, item_scores
from .modeling import load_model, load_tokenizer, resolve_device, prompt_ids, stop_ids


class QwenSpanDetector:
    def __init__(self, model_name, adapter=None, device='auto', dtype='float32', revision=None,
                 local_files_only=False, max_new_tokens=768, threshold=0.0):
        self.device = resolve_device(device)
        self.tokenizer = load_tokenizer(model_name, revision, local_files_only)
        model = load_model(model_name, dtype, revision, local_files_only)
        if adapter is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()
        model.config.use_cache = True
        self.model = model.to(self.device).eval()
        self.threshold = threshold
        stops = stop_ids(self.tokenizer)
        self.stops = set(stops)
        # Explicit config: Qwen's shipped generation_config enables sampling and
        # repetition_penalty=1.05-1.1, which penalizes copying tokens already present in
        # the prompt -- exactly what span extraction needs to do.
        self.generation = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0,
            max_new_tokens=max_new_tokens, eos_token_id=stops, pad_token_id=self.tokenizer.pad_token_id,
            temperature=None, top_p=None, top_k=None)
        # Replace the model's own config rather than passing ours to generate(): recent
        # transformers versions copy model-config values over any argument still equal to
        # the library default, which would silently restore sampling and the penalty.
        self.model.generation_config = self.generation

    def scan_detail(self, text):
        if not isinstance(text, str):
            raise ValueError('text must be a string')
        began = time.perf_counter()
        ids = torch.tensor([prompt_ids(self.tokenizer, text)], device=self.device)
        with torch.inference_mode():
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                output_logits=True, return_dict_in_generate=True)
        generated = out.sequences[0, ids.shape[1]:].tolist()
        finished = bool(generated) and generated[-1] in self.stops
        body = generated[:-1] if finished else generated
        logprobs = [torch.log_softmax(out.logits[i][0].float(), -1)[token].item() for i, token in enumerate(body)]
        raw = self.tokenizer.decode(body, skip_special_tokens=True)
        recovered = fmt.recover(text, raw)
        if not finished:
            recovered.update(spans=[], error='truncated')
        scores = item_scores(recovered, logprobs, token_char_ranges(self.tokenizer, body))
        spans = []
        if recovered['error'] is None:
            for item, score in zip(recovered['items'], scores):
                if item['status'] == fmt.ITEM_OK:
                    spans.append(dict(start=item['span'][0], end=item['span'][1], secret_type='SECRET',
                                      detector='qwen', score=min(1.0, max(0.0, score))))
        spans.sort(key=lambda s: (s['start'], s['end']))
        return dict(spans=spans, error=recovered['error'], counts=recovered['counts'],
                    canonical=recovered['canonical'], generated_tokens=len(generated),
                    latency_ms=(time.perf_counter() - began) * 1000)

    def scan_text(self, text, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        return [s for s in self.scan_detail(text)['spans'] if s['score'] >= threshold]
