# Back-end verifier: mmBERT-small on CredData-trajectory v1 (default)

This is the local back-end verifier of the two-stage detector. For each **ambiguous** detector candidate it gives P(MASK); strong candidates are masked without it. It is the default verifier: in the size comparison it had the highest system F2 and the fewest false masks ([VERIFIER_SIZE_SERIES.md](../../docs/ml_span/VERIFIER_SIZE_SERIES.md)). It is not production-ready.

## Data and training

- **Data**: the detector's candidates on CredData-trajectory v1 ([`scripts/prepare_verifier_candidates.py`](../../scripts/prepare_verifier_candidates.py)): 4,907 ambiguous candidates in train and 1,204 in validation, split by group so the verifier never trains on a group the span models were tested on. Dataset fingerprint `78b3bc38…`. The candidate data holds values and is not in this repository.
- **Training**: [`configs/verifier-creddata-mmbert-small.yaml`](../../configs/verifier-creddata-mmbert-small.yaml), full fine-tune of [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) at revision `abc32620`, window 256, batch 8, 3 epochs, seed 42, on a Colab GPU (torch 2.14.1+cu130, transformers 4.57.6).
- **Selection**: the epoch with the lowest validation loss (epoch 1). Thresholds are the highest scores that reach the recall target on validation: 0.0031 at recall 0.99 (the deployment threshold) and 0.143 at recall 0.95.

## Validation results

Test was not scored (`evaluate_test: false`).

| Level | Found / false / missed | Precision | Recall | F2 |
|---|---:|---:|---:|---:|
| Candidates, threshold at recall 0.99 | 392 / 68 / 3 | 0.852 | 0.992 | — |
| Candidates, threshold at recall 0.95 | 376 / 43 / 19 | 0.897 | 0.952 | — |
| System, strong + this verifier (1,200 gold spans) | 993 reached, 524 false masks, 207 missed | 0.655 | 0.828 | 0.786 |

Candidate-level average precision is 0.956. System metrics count candidates that overlap a gold span; they are not exact-span metrics.

## Files

- `model.safetensors`: the fine-tuned classifier (Git LFS).
- `config.json`, `tokenizer.json` (Git LFS), `tokenizer_config.json`, `special_tokens_map.json`: model and tokenizer files.
- `bundle.json`: model version, backbone and revision, selected epoch, thresholds per recall target, deployment target 0.99, dataset fingerprint.

## Using it

```python
from privai_model.verifier.runtime import CandidateVerifier
verifier = CandidateVerifier('models/verifier-creddata-mmbert-small')  # threshold: recall 0.99 on validation
scores = verifier.score(text, source, candidates)  # the detector's candidates; mask an ambiguous one if score >= verifier.threshold
```

`privai_model.inference.predict.load_model` reads the bundle directly. Pull only this folder with `git lfs pull --include "models/verifier-creddata-mmbert-small/*"`.

The base checkpoint is listed under the MIT license. Its license and notices apply to the upstream model.
