# Back-end verifier: mmBERT-base on CredData-trajectory v1 + NoseyParker-trajectory v1

The local back-end verifier retrained with NoseyParker-trajectory v1 train added: for each **ambiguous** detector candidate it gives P(MASK); strong candidates are masked without it. Same recipe as [`models/verifier-creddata-mmbert-base`](../verifier-creddata-mmbert-base/README.md); only the data differs. The default local verifier is the mmBERT-small one ([`models/verifier-creddata-np-mmbert-small`](../verifier-creddata-np-mmbert-small/README.md)). It is not production-ready.

## Data and training

- **Data**: the detector's candidates on the combined span rows (`scripts/prepare_creddata_np_traj.py`, split fingerprint `c7d0de5d…`), made by `scripts/colab_verifier.sh prepare np` with the detector the CredData-trajectory candidates came from (promptguard-claude-demoV0 `668c2c1`, CredSweeper 1.19.0, gitleaks 8.30.1). The CredData-trajectory candidates came out exactly as in that build. 4,988 ambiguous candidates in train and 1,222 in validation (NoseyParker-trajectory adds 81 and 18). Dataset fingerprint `1246451d…`. The candidate data holds values and is not in this repository.
- **Training**: [`configs/verifier-creddata-np-mmbert-base.yaml`](../../configs/verifier-creddata-np-mmbert-base.yaml), full fine-tune of [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) at revision `c5955035`, window 256, batch 8, 3 epochs, seed 42, float32, on a Colab T4.
- **Selection**: the epoch with the lowest validation loss (epoch 1: 0.589; epochs 2 and 3: 1.061, 1.006). Thresholds are the highest scores that reach the recall target on validation: 0.0400 at recall 0.99 (the deployment threshold) and 0.9616 at recall 0.95.

## Validation results

Test was not scored. The NoseyParker-trajectory test is the team's hold-out.

System level, strong candidates plus this verifier at the deployment threshold (candidates overlapping a gold span; not exact-span metrics):

| Validation set (gold spans) | Masked | False masks | Gold reached | Precision | Recall | F2 |
|---|---:|---:|---:|---:|---:|---:|
| Combined (1,323) | 1,637 | 539 | 1,098 | 0.671 | 0.830 | 0.792 |
| CredData-trajectory part (1,200) | 1,522 | 530 | 992 | 0.652 | 0.827 | 0.785 |
| NoseyParker-trajectory part (123) | 115 | 9 | 106 | 0.922 | 0.862 | 0.873 |
| (reference) CredData-only mmBERT-base verifier on CredData-trajectory (1,200) | 1,743 | 750 | 993 | 0.570 | 0.828 | 0.759 |

Candidate level at the deployment threshold: 483 ambiguous validation candidates masked, 400 of the 404 MASK candidates among them (precision 0.828, recall 0.990). On the CredData-trajectory part it masks 221 fewer candidates (220 fewer false masks) than the CredData-only mmBERT-base verifier, reaching one gold span fewer.

## Weights stored in float16

`model.safetensors` holds the trained float32 weights rounded to float16 (`scripts/half_bundle.py`), which halves the file for Git LFS. The loader still computes in float32. Rescoring the ambiguous validation candidates with the float32 and the float16 copies gave the same decisions at the deployment threshold (0 of 1,222 changed; largest P(MASK) difference 0.005). The check and the float32 source's sha256 are in `bundle.json` under `half_storage`.

## Files

- `model.safetensors`: the fine-tuned classifier, float16 (Git LFS).
- `config.json`, `tokenizer.json` (Git LFS; the same file as the other mmBERT bundles), `tokenizer_config.json`, `special_tokens_map.json`: model and tokenizer files.
- `bundle.json`: model version, backbone and revision, selected epoch, thresholds per recall target, deployment target 0.99, dataset fingerprint, `half_storage`.

## Using it

```python
from privai_model.verifier.runtime import CandidateVerifier
verifier = CandidateVerifier('models/verifier-creddata-np-mmbert-base')  # threshold: recall 0.99 on validation
scores = verifier.score(text, source, candidates)  # mask an ambiguous candidate if score >= verifier.threshold
```

`privai_model.inference.predict.load_model` reads the bundle directly. Pull only this folder with `git lfs pull --include "models/verifier-creddata-np-mmbert-base/*"`.

The base checkpoint is listed under the MIT license. Its license and notices apply to the upstream model.
