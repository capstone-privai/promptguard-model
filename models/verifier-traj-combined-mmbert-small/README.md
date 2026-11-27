# Back-end verifier: mmBERT-small on Trajectory-train-combined v1

The local back-end verifier retrained on the team's third training set, Trajectory-train-combined v1: the CredData, NoseyParker and infra-config trajectories, all with the 2026-11-27 v2 labels. For each **ambiguous** detector candidate it gives P(MASK); strong candidates are masked without it. Same recipe as [`models/verifier-creddata-mmbert-small`](../verifier-creddata-mmbert-small/README.md); only the data differs. It is not production-ready.

## Data and training

- **Data**: the detector's candidates on the span rows of [`scripts/prepare_traj_combined.py`](../../scripts/prepare_traj_combined.py) (split fingerprint `13796746…`; see [TRAJ_COMBINED.md](../../docs/ml_span/TRAJ_COMBINED.md)).
  - Candidates were made by `scripts/colab_verifier.sh prepare tc` with the detector the earlier verifiers used: promptguard-claude-demoV0 `668c2c1`, CredSweeper 1.19.0, gitleaks 8.30.1.
  - Validation holds 1,346 ambiguous candidates, 480 of them MASK. Dataset fingerprint `0d75521c…`.
  - The candidate data holds values and is not in this repository.
- **Training**: [`configs/verifier-traj-combined-mmbert-small.yaml`](../../configs/verifier-traj-combined-mmbert-small.yaml).
  - Full fine-tune of [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) at revision `abc32620`.
  - Window 256, batch 8, 3 epochs, seed 42, float32, on a Colab T4 (about 12 minutes).
- **Selection**: the epoch with the lowest validation loss, epoch 1.
  - Validation loss by epoch: 0.392, 0.520, 0.477.
  - Thresholds are the highest scores that reach the recall target on validation: 0.0011 at recall 0.99 (the deployment threshold) and 0.9896 at recall 0.95.
  - The scores are very polarized: a few MASK candidates score near zero, so reaching recall 0.99 needs a threshold near zero. The share of candidates masked (42%) is close to the earlier verifiers'.

## Validation results

Test was not scored. The NoseyParker and infra-config tests are the team's hold-outs, and the CredData test groups stay closed.

System level, strong candidates plus this verifier at the deployment threshold. Candidates count when they overlap a gold span; these are not exact-span metrics.

| Validation set (gold spans) | Masked | False masks | Gold reached | Precision | Recall | F2 |
|---|---:|---:|---:|---:|---:|---:|
| **Combined (2,091)** | **2,174** | **552** | **1,622** | **0.746** | **0.776** | **0.770** |
| CredData-trajectory part (1,196) | 1,514 | 525 | 989 | 0.653 | 0.827 | 0.785 |
| infra-config part (766) | 545 | 19 | 526 | 0.965 | 0.687 | 0.729 |
| NoseyParker-trajectory part (129) | 115 | 8 | 107 | 0.930 | 0.830 | 0.848 |
| (reference) combined, mask every candidate | 2,960 | 1,334 | 1,626 | 0.549 | 0.778 | 0.718 |
| (reference) combined, strong candidates only | 1,614 | 468 | 1,146 | 0.710 | 0.548 | 0.574 |

- **The verifier removes 782 of the 1,334 false masks** that masking every candidate would make, and loses 4 of the 1,626 gold spans the candidates reach.
- **infra-config recall is capped by the detector.** Its candidates reach only 526 of the 766 gold spans (0.687). The verifier keeps all 526 and removes 62 of the 81 false masks.
- **CredData part is close to the earlier verifier's.** The NoseyParker-trained verifier had 992 gold reached and 521 false masks, F2 0.786, on 1,200 gold spans. The labels changed (v2), so the two are not exactly comparable.

Candidate level at the deployment threshold: 560 of the 1,346 ambiguous validation candidates are masked, including 476 of the 480 MASK candidates (precision 0.850, recall 0.992).

## Weights stored in float16

`model.safetensors` holds the trained float32 weights rounded to float16 ([`scripts/half_bundle.py`](../../scripts/half_bundle.py)). That halves the file for Git LFS; the loader still computes in float32.

The ambiguous validation candidates were rescored with both copies, and the check is recorded in `bundle.json` under `half_storage`.
- The float16 copy masks 562 candidates and misses 4 MASK candidates. Compared with the training evaluation (560 masked, 4 missed), it keeps the same 476 true masks and masks 2 more KEEP candidates.
- Rescoring the float32 copy gave 559 masked and 5 missed, one decision off the training evaluation. The threshold sits among near-zero scores, where GPU rounding moves single candidates (largest P(MASK) difference between the copies: 0.007).

## Files

- `model.safetensors`: the fine-tuned classifier, float16 (Git LFS).
- `config.json`, `tokenizer.json` (Git LFS; the same file as the other mmBERT bundles), `tokenizer_config.json`, `special_tokens_map.json`: model and tokenizer files.
- `bundle.json` holds:
  - model version, backbone and revision
  - selected epoch
  - thresholds per recall target, and the deployment target 0.99
  - dataset fingerprint
  - `half_storage`

## Using it

```python
from privai_model.verifier.runtime import CandidateVerifier
verifier = CandidateVerifier('models/verifier-traj-combined-mmbert-small')  # threshold: recall 0.99 on validation
scores = verifier.score(text, source, candidates)  # mask an ambiguous candidate if score >= verifier.threshold
```

`privai_model.inference.predict.load_model` reads the bundle directly. Pull only this folder with `git lfs pull --include "models/verifier-traj-combined-mmbert-small/*"`.

The base checkpoint is listed under the MIT license. Its license and notices apply to the upstream model.
