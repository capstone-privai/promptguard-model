# Span model: mmBERT-small on CredData-trajectory v1 + NoseyParker-trajectory v1

The mmBERT-small BIO token classifier retrained with NoseyParker-trajectory v1 train added. It uses the same recipe as the CredData-trajectory mmBERT-small run in [TOKEN_SPAN_CREDDATA_TRAJ.md](../../docs/ml_span/TOKEN_SPAN_CREDDATA_TRAJ.md); only the data differs. It is not a PromptGuard default (that is `models/span-detector-v10`), and it is not production-ready.

## Data and training

- **Data**: CredData-trajectory v1, with its split unchanged, plus NoseyParker-trajectory v1 train split by rule file ([`scripts/prepare_creddata_np_traj.py`](../../scripts/prepare_creddata_np_traj.py), fingerprint `c7d0de5d…`).
  - Train has 36,658 rows with 9,582 gold spans; 482 of those spans come from NoseyParker-trajectory.
  - Validation has 5,870 rows with 1,323 gold spans: 1,200 from CredData-trajectory and 123 from NoseyParker-trajectory.
  - The data holds values and is not in this repository.
- **Training**: [`configs/token-span-creddata-np-mmbert.yaml`](../../configs/token-span-creddata-np-mmbert.yaml).
  - Full fine-tune of [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) at revision `abc32620`.
  - Windows of 512 tokens with stride 128; batch 4; learning rate 3e-5; seed 110; float16 autocast.
  - Trained on a Colab T4, about 40 minutes per epoch including the validation pass. It ran all 4 epochs.
- **Selection**: epoch and cutoff chosen on validation only, by exact-span F2 at every observed span score with precision as the tie-break. Selected: epoch 2, cutoff 0.670.

## Validation results

Exact-span counts: a prediction counts only if its start and end match a gold span exactly. Test was not scored; the NoseyParker-trajectory test is the team's hold-out.

| Validation set (gold spans) | Found / false / missed | Precision | Recall | F2 |
|---|---:|---:|---:|---:|
| **Combined (1,323)** | **1,123 / 324 / 200** | **0.776** | **0.849** | **0.833** |
| CredData-trajectory part (1,200) | 1,014 / 317 / 186 | 0.762 | 0.845 | 0.827 |
| NoseyParker-trajectory part (123) | 109 / 7 / 14 | 0.940 | 0.886 | 0.896 |
| (reference) CredData-only mmBERT-small on CredData-trajectory (1,200) | 996 / 268 / 204 | 0.788 | 0.830 | 0.821 |

The per-source rows use the cutoff chosen on the combined set.

| Epoch | Loss | Cutoff | Found / false / missed | F2 |
|---:|---:|---:|---:|---:|
| 1 | 0.0211 | 0.573 | 1,083 / 330 / 240 | 0.808 |
| **2** | 0.0071 | **0.670** | **1,123 / 324 / 200** | **0.833** |
| 3 | 0.0036 | 0.766 | 1,059 / 336 / 264 | 0.792 |
| 4 | 0.0027 | 0.800 | 1,061 / 265 / 262 | 0.802 |

## Weights stored in float16

`model.safetensors` holds the trained float32 weights rounded to float16 by [`scripts/half_bundle.py`](../../scripts/half_bundle.py), which halves the file for Git LFS. `MLSpanDetector` still loads the model in float32.

Validation was rescored with `MLSpanDetector` on both copies. The float32 bundle and the float16 bundle gave the same counts, 1,123 / 324 / 200, and these match the training run. The check is recorded in `span_bundle.json` under `half_storage`.

## Files

- `model.safetensors`: the fine-tuned BIO token classifier, float16 (Git LFS).
- `config.json`, `tokenizer.json` (Git LFS; the same file as `models/span-creddata-mmbert-base`), `tokenizer_config.json`, `special_tokens_map.json`: model and tokenizer files.
- `span_bundle.json` contains:
  - window length 512 and stride 128
  - the cutoff as `threshold`, and `best_epoch`
  - backbone and revision, and how the model was trained
  - `half_storage`

## Using it

```python
from privai_model.ml_span.runtime import MLSpanDetector
detector = MLSpanDetector('models/span-creddata-np-mmbert-small')
findings = detector.scan_text(text)
```

Pull only this folder with `git lfs pull --include "models/span-creddata-np-mmbert-small/*"`.

The base checkpoint is listed under the MIT license. Its license and notices apply to the upstream model.
