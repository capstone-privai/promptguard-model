# Step-0 pilot follow-up: model B plus our v10 train (B+v10, seed 112)

This directory holds a follow-up experiment to the team's step-0 pilot, not one of the requested models A and B. Its training data is model B's train plus all 6,304 rows of our `data/span-context-v10/train.jsonl` (3,096 positive, 3,208 hard negative). It is for comparison on the team's test sets. It is not a PromptGuard default, and it is not production-ready.

## Data and training

- **Data**: model B's data from `promptguard-data` branch `ml-train-pilot`, commit `5b812c7`, built and checked as for model B, plus our v10 train. The team's data is not in this repository.
- **Training**:
  - Recipe: [`configs/span-pilot-step0-bv10.yaml`](../../configs/span-pilot-step0-bv10.yaml), the same recipe as model B, starting from [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) at revision `abc32620`, seed 112.
  - It was trained on a Colab T4 GPU (commit `c3af5b0`).
- **Selection**: the epoch and the cutoff were chosen on the team's model-B val plus our v10 validation (807 gold spans).
  - Selected point: epoch 4, cutoff 0.70.
  - Result (found / false positive / missed): 730 / 71 / 77, precision 0.911, exact-span F2 0.906.
  - On the team's model-B val alone: 550 / 59 / 65, precision 0.903, F2 0.896. That is the same level as model B.
  - The team's test and test_ood sets were not opened.

## Files

- `model.safetensors`: fine-tuned mmBERT-small BIO span detector (stored with Git LFS).
- `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`: model and tokenizer files needed to load it.
- `span_bundle.json`: window length, stride, label mapping, the cutoff selected on validation (0.70) and the selected epoch.

## Using it

```python
from privai_model.ml_span.runtime import MLSpanDetector
detector = MLSpanDetector('models/span-pilot-step0-bv10-seed112')
findings = detector.scan_text(text)
```

The bundle is exactly as trained. Optionally, the closing-quote and start-offset cleanup can be turned on: add `"delimiter_policy": "quote-delimiters-v2"` to `span_bundle.json`, or pass `delimiter_policy='quote-delimiters-v2'`. On the team's model-B val this raised exact-span F2 from 0.896 to 0.982. The rule was designed by looking at validation errors, so that number is optimistic. Evaluate the model as trained first.

Results, seed variation and the follow-up experiments are in [the pilot report](../../docs/ml_span/PILOT_STEP0.md) and [`reports/ml_span/pilot-step0.json`](../../reports/ml_span/pilot-step0.json).

The base checkpoint is listed under the MIT license. Its license and notices apply to the upstream model.
