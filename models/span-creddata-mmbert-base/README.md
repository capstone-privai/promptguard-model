# Span model: mmBERT-base on CredData-trajectory v1

This is the mmBERT-base token classifier from the local-stage size comparison: the same data and recipe as the mmBERT-small CredData-trajectory model, with only the backbone larger (140M to 307M parameters). It is not a PromptGuard default (that is `models/span-detector-v10`), and it is not production-ready.

## Data and training

- **Data**: CredData-trajectory v1 ([`scripts/prepare_creddata_traj.py`](../../scripts/prepare_creddata_traj.py), split fingerprint `19c26b1e…`): 31,719 train rows and 4,644 validation rows with 1,200 gold spans. The data holds values and is not in this repository.
- **Training**: [`configs/token-span-creddata-mmbert-base.yaml`](../../configs/token-span-creddata-mmbert-base.yaml), full fine-tune of [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) at revision `c5955035`, batch 4, learning rate 3e-5, seed 110, float16 autocast, on a local RTX 4060 (torch 2.14.1+cu130). It stopped after epoch 3 (patience 2).
- **Selection**: epoch and cutoff on validation only (exact-span F2 at every observed span score, precision tie-break). Selected: epoch 1, cutoff 0.752.

## Validation results

| Model | Found / false / missed | Precision | Recall | Exact-span F2 |
|---|---:|---:|---:|---:|
| **This bundle (mmBERT-base)** | 1,029 / 353 / 171 | 0.745 | 0.858 | 0.832 |
| (reference) mmBERT-small, same recipe | 996 / 268 / 204 | 0.788 | 0.830 | 0.821 |

Test was not scored: CredData-trajectory test had already been opened for earlier models, so the size comparison uses validation only. Details are in [TOKEN_SPAN_CREDDATA_TRAJ.md](../../docs/ml_span/TOKEN_SPAN_CREDDATA_TRAJ.md#크기-비교-mmbert-base-validation만).

## How this bundle was made

The training run kept one model per epoch. [`scripts/export_token_span_bundle.py`](../../scripts/export_token_span_bundle.py) wrote the epoch and cutoff that `selection.json` chose as this bundle (float32), then rescored validation with `MLSpanDetector` on a Colab T4. The counts matched training exactly (`export_check` in `span_bundle.json`).

## Files

- `model.safetensors`: the fine-tuned BIO token classifier (Git LFS).
- `config.json`, `tokenizer.json` (Git LFS), `tokenizer_config.json`, `special_tokens_map.json`: model and tokenizer files.
- `span_bundle.json`: window length 512, stride 128, the cutoff as `threshold`, `best_epoch`, backbone and revision, how it was trained, and the validation counts with the export check.

## Using it

```python
from privai_model.ml_span.runtime import MLSpanDetector
detector = MLSpanDetector('models/span-creddata-mmbert-base')
findings = detector.scan_text(text)
```

Pull only this folder with `git lfs pull --include "models/span-creddata-mmbert-base/*"`.

The base checkpoint is listed under the MIT license. Its license and notices apply to the upstream model.
