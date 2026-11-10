# Team step-0 pilot: model A (seed 112)

This directory holds a second training run of the team's step-0 pilot model A. Its training data is the hand templates plus privesc. It differs from `models/span-pilot-step0-a-seed110` only in the random seed. It is a pilot model for the team's evaluation, not a PromptGuard default, and it is not production-ready.

## Data and training

- **Data**: `promptguard-data` branch `ml-train-pilot`, commit `5b812c7`. The data was built with the team's `build_ml_handoff.sh`, and all 30 files matched the team's sha256 values.
  - Model A uses 1,108 train sessions and 80 val sessions.
  - No rows were dropped.
  - The data itself is not in this repository.
- **Training**:
  - Recipe: [`configs/span-pilot-step0-a.yaml`](../../configs/span-pilot-step0-a.yaml), starting from [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) at revision `abc32620`, seed 112.
  - It was trained on a Colab T4 GPU.
- **Selection**: the epoch and the cutoff were chosen on the team's model-A val only (435 gold spans).
  - Selected point: epoch 3, cutoff 0.98.
  - Result (found / false positive / missed): 389 / 46 / 46, precision 0.894, exact-span F2 0.894.
  - The team's test and test_ood sets were not opened.

## Files

- `model.safetensors`: fine-tuned mmBERT-small BIO span detector (stored with Git LFS).
- `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`: model and tokenizer files needed to load it.
- `span_bundle.json`: window length, stride, label mapping, the cutoff selected on validation (0.98) and the selected epoch.

## Using it

```python
from privai_model.ml_span.runtime import MLSpanDetector
detector = MLSpanDetector('models/span-pilot-step0-a-seed112')
findings = detector.scan_text(text)
```

The bundle is exactly as trained. Optionally, the closing-quote and start-offset cleanup can be turned on: add `"delimiter_policy": "quote-delimiters-v2"` to `span_bundle.json`, or pass `delimiter_policy='quote-delimiters-v2'`. On the team's model-A val this raised exact-span F2 from 0.894 to 0.998. The rule was designed by looking at validation errors, so that number is optimistic. Evaluate the model as trained first.

Results, seed variation and the follow-up experiments are in [the pilot report](../../docs/ml_span/PILOT_STEP0.md) and [`reports/ml_span/pilot-step0.json`](../../reports/ml_span/pilot-step0.json).

The base checkpoint is listed under the MIT license. Its license and notices apply to the upstream model.
