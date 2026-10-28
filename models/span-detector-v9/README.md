# PromptGuard ML Span Detector v9

This directory contains the selected v9 token-span model bundle. It was trained on synthetic credential examples only; it has not been evaluated on the deployed PromptGuard ensemble and is not production-ready.

## Contents

- `model.safetensors`: fine-tuned mmBERT-small BIO span detector.
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`: tokenizer files required for local loading.
- `boundary.pt`: character-boundary refiner bundled with the selected checkpoint.
- `span_bundle.json`: maximum sequence length, stride, label mapping, and selected threshold.

The score threshold (`0.9996943789993322`) was selected on synthetic validation data to meet the v8 reference precision/recall constraints. It is not a calibrated probability and may be sensitive to distribution changes. See [v9 evaluation](../../docs/ml_span/ML_C_CONTEXT_V9.md) for results, failed regression gates, and limitations. Synthetic data and split manifests are in `data/span-context-v9/`; prior cumulative training sets are documented under `docs/ml_span/`.

The base checkpoint is [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small), which is listed under the MIT license. Its license and notices apply to the upstream model; this fine-tune adds PromptGuard-specific synthetic training.
