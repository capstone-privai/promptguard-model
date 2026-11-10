# Team step-0 pilot: model B (seed 110)

This directory holds the token-span model trained for the team's step-0 pilot as model B. Its training data is the hand templates, the LLM templates and privesc. It is a pilot model for the team's evaluation, not a PromptGuard default, and it is not production-ready.

## Data and training

- **Data**: `promptguard-data` branch `ml-train-pilot`, commit `5b812c7`. The data was built with the team's `build_ml_handoff.sh`, and all 30 files matched the team's sha256 values.
  - Model B uses 1,693 train sessions and 155 val sessions.
  - No rows were dropped.
  - The data itself is not in this repository.
- **Training**:
  - Recipe: [`configs/span-pilot-step0-b.yaml`](../../configs/span-pilot-step0-b.yaml), starting from [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) at revision `abc32620`, seed 110.
  - Model A used the same recipe; only the data differs.
- **Selection**: the epoch and the cutoff were chosen on the team's model-B val only (615 gold spans).
  - Selected point: epoch 3, cutoff 0.90.
  - Result (found / false positive / missed): 551 / 55 / 64, precision 0.909, exact-span F2 0.899.
  - The team's test and test_ood sets were not opened.

## Files

- `model.safetensors`: fine-tuned mmBERT-small BIO span detector (stored with Git LFS).
- `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`: model and tokenizer files needed to load it.
- `span_bundle.json`: window length, stride, label mapping, the cutoff selected on validation (0.90) and the selected epoch.

## Using it

```python
from privai_model.ml_span.runtime import MLSpanDetector
detector = MLSpanDetector('models/span-pilot-step0-b-seed110')
findings = detector.scan_text(text)
```

The bundle is exactly as trained. Optionally, the closing-quote and start-offset cleanup can be turned on: add `"delimiter_policy": "quote-delimiters-v2"` to `span_bundle.json`, or pass `delimiter_policy='quote-delimiters-v2'`. On the team's model-B val this raised exact-span F2 from 0.899 to 0.985. The rule was designed by looking at validation errors, so that number is optimistic. Evaluate the model as trained first.

Results, seed variation and the follow-up experiments are in [the pilot report](../../docs/ml_span/PILOT_STEP0.md) and [`reports/ml_span/pilot-step0.json`](../../reports/ml_span/pilot-step0.json).

The base checkpoint is listed under the MIT license. Its license and notices apply to the upstream model.
