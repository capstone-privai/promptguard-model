# Span model: mmBERT-base on CredData-trajectory v1 + NoseyParker-trajectory v1

The mmBERT-base BIO token classifier, retrained with NoseyParker-trajectory v1 train added. It uses the same recipe as the CredData-trajectory mmBERT-base run in [TOKEN_SPAN_CREDDATA_TRAJ.md](../../docs/ml_span/TOKEN_SPAN_CREDDATA_TRAJ.md); only the data differs. It is not a PromptGuard default (that is `models/span-detector-v10`), and it is not production-ready.

## Data and training

- **Data**: CredData-trajectory v1, with its split unchanged, plus NoseyParker-trajectory v1 train split by rule file ([`scripts/prepare_creddata_np_traj.py`](../../scripts/prepare_creddata_np_traj.py), fingerprint `c7d0de5d…`).
  - Train has 36,658 rows with 9,582 gold spans; 482 of those spans come from NoseyParker-trajectory.
  - Validation has 5,870 rows with 1,323 gold spans: 1,200 from CredData-trajectory and 123 from NoseyParker-trajectory.
  - The data holds values and is not in this repository.
- **Training**: [`configs/token-span-creddata-np-mmbert-base.yaml`](../../configs/token-span-creddata-np-mmbert-base.yaml).
  - Full fine-tune of [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) (307M parameters) at revision `c5955035`.
  - Windows of 512 tokens with stride 128; batch 4; learning rate 3e-5; seed 110; float16 autocast.
  - Trained on a Colab T4, about 76 minutes per epoch including the validation pass. It ran all 4 epochs.
- **Selection**: epoch and cutoff chosen on validation only, by exact-span F2 at every observed span score with precision as the tie-break. Selected: epoch 3, cutoff 0.833.

## Validation results

Exact-span counts: a prediction counts only if its start and end match a gold span exactly. Test was not scored; the NoseyParker-trajectory test is the team's hold-out.

| Validation set (gold spans) | Found / false / missed | Precision | Recall | F2 |
|---|---:|---:|---:|---:|
| **Combined (1,323)** | **1,146 / 301 / 177** | **0.792** | **0.866** | **0.850** |
| CredData-trajectory part (1,200) | 1,033 / 292 / 167 | 0.780 | 0.861 | 0.843 |
| NoseyParker-trajectory part (123) | 113 / 9 / 10 | 0.926 | 0.919 | 0.920 |
| (reference) CredData-only mmBERT-base on CredData-trajectory (1,200) | 1,029 / 353 / 171 | 0.745 | 0.858 | 0.832 |
| (reference) the NP mmBERT-small, combined (1,323) | 1,123 / 324 / 200 | 0.776 | 0.849 | 0.833 |

The per-source rows use the cutoff chosen on the combined set.

| Epoch | Cutoff | Found / false / missed | F2 |
|---:|---:|---:|---:|
| 1 | 0.508 | 1,010 / 251 / 313 | 0.771 |
| 2 | 0.638 | 1,026 / 303 / 297 | 0.775 |
| **3** | **0.833** | **1,146 / 301 / 177** | **0.850** |
| 4 | 0.975 | 1,122 / 270 / 201 | 0.839 |

## Weights stored in float16

`model.safetensors` holds the trained float32 weights rounded to float16 by [`scripts/half_bundle.py`](../../scripts/half_bundle.py), which halves the file for Git LFS. `MLSpanDetector` still loads the model in float32.

Validation was rescored with `MLSpanDetector` on both copies. The float32 bundle and the float16 bundle gave the same counts, 1,146 / 301 / 177, and these match the training run. The check is recorded in `span_bundle.json` under `half_storage`.

## Files

- `model.safetensors`: the fine-tuned BIO token classifier, float16 (Git LFS).
- `config.json`, `tokenizer.json` (Git LFS; the same file as `models/span-creddata-mmbert-base`), `tokenizer_config.json`, `special_tokens_map.json`: model and tokenizer files.
- `span_bundle.json` holds:
  - window length 512 and stride 128
  - the cutoff as `threshold`, and `best_epoch`
  - backbone and revision, and how the model was trained
  - `half_storage`

## Using it

```python
from privai_model.ml_span.runtime import MLSpanDetector
detector = MLSpanDetector('models/span-creddata-np-mmbert-base')
findings = detector.scan_text(text)
```

Pull only this folder with `git lfs pull --include "models/span-creddata-np-mmbert-base/*"`.

The base checkpoint is listed under the MIT license. Its license and notices apply to the upstream model.
