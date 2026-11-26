# Back-end verifier: mmBERT-small on CredData-trajectory v1 + NoseyParker-trajectory v1

The local back-end verifier retrained with NoseyParker-trajectory v1 train added: for each **ambiguous** detector candidate it gives P(MASK); strong candidates are masked without it. Same recipe as [`models/verifier-creddata-mmbert-small`](../verifier-creddata-mmbert-small/README.md); only the data differs. It is not production-ready.

## Data and training

- **Data**: the detector's candidates on the combined span rows (`scripts/prepare_creddata_np_traj.py`, split fingerprint `c7d0de5d…`), made by `scripts/colab_verifier.sh prepare np` with the detector the CredData-trajectory candidates came from (promptguard-claude-demoV0 `668c2c1`, CredSweeper 1.19.0, gitleaks 8.30.1). The CredData-trajectory candidates came out exactly as in that build. 4,988 ambiguous candidates in train and 1,222 in validation (NoseyParker-trajectory adds 81 and 18). Dataset fingerprint `1246451d…`. The candidate data holds values and is not in this repository.
- **Training**: [`configs/verifier-creddata-np-mmbert-small.yaml`](../../configs/verifier-creddata-np-mmbert-small.yaml), full fine-tune of [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) at revision `abc32620`, window 256, batch 8, 3 epochs, seed 42, float32, on a Colab T4.
- **Selection**: the epoch with the lowest validation loss (epoch 1: 0.682; epochs 2 and 3: 0.822, 0.701). Thresholds are the highest scores that reach the recall target on validation: 0.9808 at recall 0.99 (the deployment threshold) and 0.9980 at recall 0.95.

## Validation results

Test was not scored. The NoseyParker-trajectory test is the team's hold-out.

System level, strong candidates plus this verifier at the deployment threshold (candidates overlapping a gold span; not exact-span metrics):

| Validation set (gold spans) | Masked | False masks | Gold reached | Precision | Recall | F2 |
|---|---:|---:|---:|---:|---:|---:|
| Combined (1,323) | 1,628 | 530 | 1,098 | 0.674 | 0.830 | 0.793 |
| CredData-trajectory part (1,200) | 1,513 | 521 | 992 | 0.656 | 0.827 | 0.786 |
| NoseyParker-trajectory part (123) | 115 | 9 | 106 | 0.922 | 0.862 | 0.873 |
| (reference) CredData-only verifier on CredData-trajectory (1,200) | 1,517 | 524 | 993 | 0.655 | 0.828 | 0.786 |

Candidate level at the deployment threshold: 474 ambiguous validation candidates masked, 400 of the 404 MASK candidates among them (precision 0.844, recall 0.990).

## Weights stored in float16

`model.safetensors` holds the trained float32 weights rounded to float16 (`scripts/half_bundle.py`), which halves the file for Git LFS. The loader still computes in float32. Rescoring the ambiguous validation candidates with the float32 and the float16 copies gave the same decisions at the deployment threshold (0 of 1,222 changed; largest P(MASK) difference 0.004). The check and the float32 source's sha256 are in `bundle.json` under `half_storage`.

## Files

- `model.safetensors`: the fine-tuned classifier, float16 (Git LFS).
- `config.json`, `tokenizer.json` (Git LFS; the same file as the other mmBERT bundles), `tokenizer_config.json`, `special_tokens_map.json`: model and tokenizer files.
- `bundle.json`: model version, backbone and revision, selected epoch, thresholds per recall target, deployment target 0.99, dataset fingerprint, `half_storage`.

## Using it

```python
from privai_model.verifier.runtime import CandidateVerifier
verifier = CandidateVerifier('models/verifier-creddata-np-mmbert-small')  # threshold: recall 0.99 on validation
scores = verifier.score(text, source, candidates)  # mask an ambiguous candidate if score >= verifier.threshold
```

`privai_model.inference.predict.load_model` reads the bundle directly. Pull only this folder with `git lfs pull --include "models/verifier-creddata-np-mmbert-small/*"`.

The base checkpoint is listed under the MIT license. Its license and notices apply to the upstream model.
