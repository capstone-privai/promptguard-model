# Step-0 pilot follow-up: model B plus CredData-trajectory v1 train (B+CredData, seed 112)

This directory holds a follow-up experiment to the team's step-0 pilot, not one of the requested models A and B. Its training data is model B's train plus all 31,719 rows of CredData-trajectory v1 train (9,100 gold spans; real repositories' credentials placed in agent trajectories, see [the CredData-trajectory report](../../docs/ml_span/TOKEN_SPAN_CREDDATA_TRAJ.md)). It is for comparison on the team's test sets. It is not a PromptGuard default, and it is not production-ready.

## Data and training

- **Data**: model B's data from `promptguard-data` branch `ml-train-pilot`, commit `5b812c7`, built and checked as for model B, plus CredData-trajectory v1 train (`scripts/prepare_pilot_plus_ours.py bcred`, data fingerprint `95fa9c2d`). Neither dataset is in this repository.
- **Training**:
  - Recipe: [`configs/span-pilot-step0-bcred.yaml`](../../configs/span-pilot-step0-bcred.yaml), the same recipe as model B+v10, starting from [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) at revision `abc32620`, seed 112.
  - It was trained on a Colab T4 GPU (commit `5984057`).
  - **The run was interrupted.** The Colab runtime ended after epoch 5 of at most 8. Epochs 4 and 5 scored below epoch 3, so with patience 3 the run would have stopped after epoch 6 and kept epoch 3 unless epoch 6 scored higher; epoch 6 was not trained. `span_bundle.json` records this (`interrupted`).
- **Selection**: the epoch and the cutoff were chosen on the team's model-B val plus CredData-trajectory validation (1,815 gold spans).
  - Selected point: epoch 3, cutoff 0.95.
  - Result (found / false positive / missed): 1,574 / 349 / 241, precision 0.819, exact-span F2 0.857.
  - On the team's model-B val alone: 547 / 56 / 68, precision 0.907, F2 0.893. That is the same level as models B and B+v10.
  - On CredData-trajectory validation alone: 1,027 / 293 / 173, precision 0.778, F2 0.839.
  - On our own synthetic development sets it misses more secrets than model B (for example 75 against 16 on fresh_test), though with fewer false positives; it did not learn formats it was not trained on.
  - The team's test and test_ood sets were not opened.

## Files

- `model.safetensors`: fine-tuned mmBERT-small BIO span detector (stored with Git LFS).
- `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`: model and tokenizer files needed to load it.
- `span_bundle.json`: window length, stride, label mapping, the cutoff selected on validation (0.95), the selected epoch and the interruption record.

## Using it

```python
from privai_model.ml_span.runtime import MLSpanDetector
detector = MLSpanDetector('models/span-pilot-step0-bcred-seed112')
findings = detector.scan_text(text)
```

The bundle is exactly as trained. The optional quote cleanup (`"delimiter_policy": "quote-delimiters-v2"` in `span_bundle.json`) is described in [the pilot report](../../docs/ml_span/PILOT_STEP0.md); evaluate the model as trained first.

Results, the comparison with B+v10 and the other follow-up experiments are in [the pilot report](../../docs/ml_span/PILOT_STEP0.md) and [`reports/ml_span/pilot-step0.json`](../../reports/ml_span/pilot-step0.json).

The base checkpoint is listed under the MIT license. Its license and notices apply to the upstream model.
