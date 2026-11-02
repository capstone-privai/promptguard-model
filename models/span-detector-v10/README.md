# PromptGuard ML Span Detector v10

This directory contains the v10 token-span model bundle, the current default mmBERT span detector. It was trained on synthetic credential examples only; it has not been evaluated on the deployed PromptGuard ensemble and is not production-ready.

## Contents

- `model.safetensors`: fine-tuned mmBERT-small BIO span detector (v10 warm run, epoch 3).
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`: tokenizer files required for local loading.
- `boundary.pt`: character-boundary refiner (boundary-v3, the same refiner as v9).
- `span_bundle.json`: maximum sequence length, stride, label mapping, selected threshold, runtime candidate filter and selection record.

## How it was selected

v10 warm-starts v9 and retrains on the de-duplicated union of every earlier training split (6,304 rows) at learning rate 5e-6. The boundary refiner and the score threshold (`0.9856988658706943`) were selected on validation only, constrained to be no worse than v9 in precision, recall and F2. The threshold is not a calibrated probability.

The run chosen on validation was a from-scratch variant (base). It failed the predeclared gates, with a recall collapse on early datasets. The warm run was then checked on a new independent hold-out, `data/span-holdout-v11`, using a decision rule fixed beforehand. Against v9 it raised F2 from 0.777 to 0.853 and precision from 0.716 to 0.746.

Warm passes 10 of the 13 predeclared v10 gates, the same count as v9. It fails v6 retention (FP 14 > 3), v7 retention (FP 12 > 9) and mixed trailing-role recovery (FP 5 > 1). It was made the default by an explicit decision to accept those gate failures. Known weaknesses include public IDs next to secrets in long logs, and secrets whose role is shown only by a CSV/TSV header. See [v10 evaluation](../../docs/ml_span/ML_C_CONTEXT_V10.md).

The previous bundle stays in `models/span-detector-v9` for comparison.

## Grammatical-word candidate policy

Like v9, the bundle enables `contextual-function-words-v1` in the Python `MLSpanDetector` runtime. Direct Transformers inference does not apply this policy. See [policy](../../docs/ml_span/V9_FUNCTION_WORD_FIX.md).

The base checkpoint is [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small), which is listed under the MIT license. Its license and notices apply to the upstream model; this fine-tune adds PromptGuard-specific synthetic training.
