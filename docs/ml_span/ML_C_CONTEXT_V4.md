# Context v4 and learned character boundaries

## Purpose and evidence

The eight unresolved original development misses were all the `header` family (`Authorization: Bearer …`). That finding is development evidence, not a new hold-out result. The original examples/values were not put into training. New HTTP client, gRPC, proxy-log and shell-authentication contexts were independently authored.

This experiment also addresses mixed public/secret values in one input, multiple secrets, repeated values, multiline material, Korean suffixes, Unicode and long inputs. These are synthetic challenge dimensions; they are not asserted to be verified categories from the public PromptGuard page.

## Data and leakage control

Thirty new context families have fixed pre-training assignments: 18 train, six validation, six test. Each family has 24 pairs. A pair uses identical generated values in private authentication and public-reference contexts. Public fields inside positive rows remain unlabelled. Value formats include alphanumeric, punctuation, hex, base64, URL-safe, dotted, numeric, Unicode and multiline strings. Long contexts vary position and exceed one 512-token window.

Prior training rows are replayed: total Train 2,184 rows / 1,216 gold spans; Validation 288 rows / 168 gold; fresh Test 288 rows / 192 gold. Across the entire checkpoint lineage, all prior validation and test families remain excluded from training. Group checks cover family, template, origin, value family, text, canonical template and value hashes. Public page value/template exclusion hashes apply before a whole semantic pair is retained. These hash checks do not prove semantic independence from every public template.

Previous model tests are explicitly development regression checks. The new test must also become development data if later changes are chosen after examining its results.

## Model training

Warm start: `runs/span-mmbert-context-v3/model`. Learning rate 3e-6, up to three epochs, CPU four threads, class weights 1/2/2, 512-token windows and 128-token overlap. Epoch and span-confidence threshold are selected by validation F2. No performance on the deployed PromptGuard ensemble is measured; rules use the local CredSweeper adapter and recorded version/config hash.

Token alignment diagnostics found 68 of 1,216 training gold spans and 22 of 168 validation gold spans that token-level BIO cannot represent exactly in one window. Examples include a base64 value's final `==` sharing a token with following quotes/comma, or punctuation attached to a Korean suffix. This is not evidence of a missing Secret by itself.

## Optional character-boundary model

A small learned model sees Unicode characters/categories around an already detected candidate's start/end. It predicts an inward trim of 0–8 code points, and cannot create or expand candidates, change confidence, or alter Rule findings. Invalid contractions leave the original candidate unchanged. This is an auxiliary ML model, not a new Rule Detector.

It trains on 2,432 token-expanded gold boundaries from training data only. Epoch/trim confidence selection uses validation boundary correctness with a penalty for incorrect trims. It must then strictly improve full-detector exact validation F2 without lowering precision before a combined bundle is created. Final test access is not used by this acceptance step.

The tokenizer's segmentation is preserved. Transformers' installed local-file Mistral warning heuristic also triggers for the saved ModernBERT config version; applying its suggested Mistral tokenizer rewrite would change the inherited tokenizer. This experiment does not apply that rewrite or claim the warning itself has been resolved.

## Reproduction

Run from repository root; output directories must not already exist:

```sh
PYTHONPATH=src .venv/bin/python scripts/prepare_context_v4.py
PYTHONPATH=src .venv/bin/python scripts/span_pipeline.py train --config configs/span-mmbert-context-v4.yaml
PYTHONPATH=src .venv/bin/python scripts/train_span_boundaries.py
PYTHONPATH=src .venv/bin/python scripts/validate_span_boundaries.py
```

If the boundary validation gate accepts the combined bundle:

```sh
PYTHONPATH=src .venv/bin/python scripts/compare_span_checkpoints.py \
  --previous runs/span-mmbert-context-v3/model \
  --candidate runs/span-mmbert-context-v4-boundary/model \
  --unrefined runs/span-mmbert-context-v4/model \
  --data-dir data/span-context-v4 \
  --output reports/ml_span/context-v4-comparison.json
```

Otherwise compare the unrefined v4 checkpoint as candidate. Both versions remain experiments. Model weights and dataset JSONL are ignored by Git. Reports contain aggregates and error IDs/coordinates, not source text or credential values.

## V5 validation-driven follow-up

Before reading final test results, v4 validation showed poor transfer despite low training loss. A second candidate adds 864 fresh synthetic training rows with context-only paraphrases (`password`/`passphrase`, `secret`/`credential`, request/correlation and Korean equivalents). Registered value spans are never modified, and annotations are rebuilt from exact original value segments. The training families and held-out family assignment are preserved. Total training size is 3,048 records. This is iterative validation development, not another independent validation claim.

V5 warm-starts from the validation-selected v4 checkpoint at learning rate 2e-6 for up to two epochs. Length-grouped batches shuffle equal-length examples and batch order, retain every training row, and reduce padding of short examples to the longest input. V4 uses its original random batching; the v5 configuration explicitly opts into grouping.

```sh
PYTHONPATH=src .venv/bin/python scripts/prepare_context_v5.py
PYTHONPATH=src .venv/bin/python scripts/span_pipeline.py train --config configs/span-mmbert-context-v5.yaml
PYTHONPATH=src .venv/bin/python scripts/validate_span_boundaries.py \
  --checkpoint runs/span-mmbert-context-v5-b16/model \
  --data-dir data/span-context-v5 \
  --output runs/span-mmbert-context-v5-boundary/model \
  --report reports/ml_span/context-v5-boundary-validation.json
```

The boundary model remains trained on the v4 training subset. Both detectors use the same validation-gated boundary weights, preventing test-based changes to the correction stage.

A separate 96-row, 12-family negative challenge covers environment references, placeholders, redacted values, empty fields, schemas, vault references and public authentication documentation. It is not used for training, checkpoint selection or threshold selection. Prepare with `scripts/prepare_negative_challenge.py`. It tests false positives only and cannot estimate recall.

The first v5 run with batch size four was interrupted before completing an epoch and is excluded from performance claims. The final v5-b16 run restarts from v4 with batch size 16 and the same fixed data; the interrupted run configuration/log is retained locally.
