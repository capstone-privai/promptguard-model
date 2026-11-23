# Back-end verifier: mmBERT-base on CredData-trajectory v1 (comparison, not the default)

This is the local back-end verifier trained with a larger backbone: the same candidates and recipe as the default [`verifier-creddata-mmbert-small`](../verifier-creddata-mmbert-small/), with mmBERT-base (307M parameters) in place of mmBERT-small (140M). **It is worse than mmBERT-small on validation and is kept for comparison.** It is not production-ready.

## Data and training

- **Data**: the detector's candidates on CredData-trajectory v1 ([`scripts/prepare_verifier_candidates.py`](../../scripts/prepare_verifier_candidates.py)): 4,907 ambiguous candidates in train and 1,204 in validation, split by group. Dataset fingerprint `78b3bc38…`, the same as the small verifier. The candidate data holds values and is not in this repository.
- **Training**: [`configs/verifier-creddata-mmbert-base.yaml`](../../configs/verifier-creddata-mmbert-base.yaml), full fine-tune of [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) at revision `c5955035`, window 256, batch 8, learning rate 2e-5, 3 epochs, seed 42, float32, on a Colab T4 (commit `a7589a9`, torch 2.14.1+cu130, transformers 4.57.6). About 12 minutes per epoch.
- **Selection**: the epoch with the lowest validation loss. Validation loss was 0.673, 0.840 and 1.647 for epochs 1 to 3, so epoch 1. Thresholds: 0.0011 at recall 0.99 (the deployment threshold) and 0.420 at recall 0.95.

## Validation results

Test was not scored (`evaluate_test: false`).

| Level | This bundle (mmBERT-base) | mmBERT-small (default) |
|---|---:|---:|
| Candidates at recall 0.99: found / false / missed, precision | 392 / 294 / 3, 0.571 | 392 / 68 / 3, 0.852 |
| Candidates at recall 0.95: found / false / missed, precision | 376 / 125 / 19, 0.750 | 376 / 43 / 19, 0.897 |
| Candidate average precision | 0.849 | 0.956 |
| System, strong + verifier: masked / false masks | 1,743 / 750 | 1,517 / 524 |
| System precision / recall / F2 (1,200 gold spans) | 0.570 / 0.828 / 0.759 | 0.655 / 0.828 / 0.786 |

The recipe is the small model's, unchanged, and there is one seed. A learning rate or epoch count tuned for mmBERT-base on validation might do better. The comparison with every verifier size is in [VERIFIER_SIZE_SERIES.md](../../docs/ml_span/VERIFIER_SIZE_SERIES.md).

## Files

- `model.safetensors`: the fine-tuned classifier (Git LFS).
- `config.json`, `tokenizer.json` (Git LFS), `tokenizer_config.json`, `special_tokens_map.json`: model and tokenizer files.
- `bundle.json`: model version, backbone and resolved revision, selected epoch, thresholds per recall target, deployment target 0.99, dataset fingerprint.

## Using it

```python
from privai_model.verifier.runtime import CandidateVerifier
verifier = CandidateVerifier('models/verifier-creddata-mmbert-base')  # threshold: recall 0.99 on validation
scores = verifier.score(text, source, candidates)
```

Pull only this folder with `git lfs pull --include "models/verifier-creddata-mmbert-base/*"`.

The base checkpoint is listed under the MIT license. Its license and notices apply to the upstream model.
