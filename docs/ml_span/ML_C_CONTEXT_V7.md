# Context v7: semantic role and variable multiline training

## Design and provenance

This experiment follows development analysis of v4/v6 failures. At threshold zero, some multiline credentials had only a suffix predicted and some values explained by following text had no predictions. A public password-policy reference also produced a high-confidence false positive. These observations motivate context diversity; they are not newly discovered public benchmark cases.

The new generator authors 24 independent families, assigned to 16 training, four validation and four test families before training. It adds 1,024 paired synthetic training rows to the 3,432 historical training rows. Training has 4,456 rows and 2,536 gold spans. Each validation/test split has 256 rows and 160 gold spans. The new pairs share identically shaped values but differ in whether those values authorize private access or identify public documentation. New contexts include role explanations after values, Korean/English mixtures, two-secret inputs and multiline blocks with two to five lines of 12–48 characters each.

The new script checks public-example value/template hashes, group/value/template separation within splits, and separation of all cumulative training from historical v1/v3/v4/v6 validation/test rows. Earlier tests are development evaluations. No public example text or credential is used as a training record. Exact hash exclusions do not prove absence of semantic similarity or cover inaccessible public data.

The data retains existing Rule-hit positives and negatives. Training is not restricted to Rule misses. `context-v7-dataset.json` records class/source/format counts. The generator supports explicitly passed family lists without mutating its global default, and a regression test verifies pair integrity, variable multiline offsets and split separation.

## Training and selection

The span model warm-starts the raw v6 checkpoint; maximum three epochs, batch 32, length grouping, learning rate 2e-6, CPU four threads, seed 94. Validation exact-span F2 selects the epoch and score threshold. Historical development and fresh-test results do not select the epoch or threshold.

A second character boundary model is trained on v7 training labels only. Its character-level validation score penalizes incorrect trims; a separate full-detector validation gate must improve F2 without reducing precision before a bundled model is created. The boundary model can only trim existing candidates, and cannot invent, extend or join candidates. Train token alignment reports 167/2,536 gold spans with boundary/window representation limitations; validation reports 25/160. These are representation diagnostics, not counts of confirmed model misses.

## Reproduction

```sh
.venv/bin/python scripts/prepare_context_v7.py
.venv/bin/python scripts/span_pipeline.py train --config configs/span-mmbert-context-v7.yaml
.venv/bin/python scripts/train_span_boundaries.py --output runs/span-boundary-v2 --data-dir data/span-context-v7 --tokenizer runs/span-mmbert-context-v6/model
.venv/bin/python scripts/validate_span_boundaries.py --checkpoint runs/span-mmbert-context-v7/model --data-dir data/span-context-v7 --output runs/span-mmbert-context-v7-boundary/model --boundary-dir runs/span-boundary-v2 --report reports/ml_span/context-v7-boundary-validation.json
```

The generators/trainers refuse to overwrite existing output directories. Checkpoints and generated input records stay local and ignored by Git. Reports include aggregate metrics and error IDs/coordinates, not secret values. Results are synthetic and use the local Rule proxy; deployed PromptGuard ensemble performance remains unmeasured. Four new test families provide limited evidence of generalization. Latency is sequential local CPU inference and input-length dependent.

Run the frozen comparison (including historical development regression checks):

```sh
.venv/bin/python scripts/compare_span_checkpoints.py --previous runs/span-mmbert-context-v6-boundary/model --candidate runs/span-mmbert-context-v7-boundary/model --unrefined runs/span-mmbert-context-v7/model --data-dir data/span-context-v7 --output reports/ml_span/context-v7-comparison.json --development v4_development=data/span-context-v4/test.jsonl --development v6_development=data/span-context-v6/test.jsonl
```

`context-v7-training.json` preserves the epoch history, configuration, alignment diagnostics, synthetic dataset manifest and checkpoint file SHA-256 hashes. The span model completed all three epochs and selected epoch 3 at score threshold 0.95. Epoch 2 had 75 TP / 66 FP on validation; epoch 3 had 74 TP / 56 FP, a very small F2 improvement (0.480154 to 0.480519). This is a precision/recall tradeoff, not evidence that every additional epoch improves recall. Character training completed 40 epochs and selected epoch 1: 315/320 correct validation boundaries with zero wrong trims. Full-detector validation accepted the head: 74 TP / 56 FP / 86 FN became 93 TP / 37 FP / 67 FN.

## Frozen comparison results

Counts are exact-span TP / FP / FN. Each row compares the same inputs; only `fresh_test` is the new held-out test for this iteration.

| Dataset | V6 boundary | V7 boundary | V7 raw |
|---|---:|---:|---:|
| validation | 74 / 87 / 86 | 93 / 37 / 67 | 74 / 56 / 86 |
| fresh_test | 25 / 63 / 135 | 85 / 9 / 75 | 81 / 13 / 79 |
| v3_development | 128 / 0 / 0 | 128 / 0 / 0 | 128 / 0 / 0 |
| v1_development | 48 / 0 / 0 | 48 / 0 / 0 | 48 / 0 / 0 |
| original_development | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 |
| negative_challenge | 0 / 7 / 0 | 0 / 7 / 0 | 0 / 7 / 0 |
| v4_development | 150 / 58 / 42 | 143 / 37 / 49 | 132 / 48 / 60 |
| v6_development | 54 / 20 / 42 | 67 / 3 / 29 | 61 / 9 / 35 |

### Fresh synthetic test: Rule, ML and union

| Detector | TP | FP | FN | Exact precision | Exact recall |
|---|---:|---:|---:|---:|---:|
| rule_only | 32 | 32 | 128 | 50.00% | 20.00% |
| ml_only | 85 | 9 | 75 | 90.43% | 53.12% |
| candidate_union | 96 | 41 | 64 | 70.07% | 60.00% |

Rule misses: 128; ML adds 64 exact true positives, recovering 50.00%, and induces 9 additional false positives. V6 recovered 25/128 misses and induced 63 FP on these same inputs. V7 ML exact precision is 90.43%, but exact recall remains only 53.12%; the union reaches 60.00% recall.

Mean ML latency: V6 22.48 ms/input; V7 22.71 ms/input (p95 37.08 ms/input). These timings do not establish a significant speed difference. Rule latency is reported separately in the JSON; a production end-to-end union runtime is not measured.

### Remaining failures and adoption limits

| New test family | TP | FP | FN |
|---|---:|---:|---:|
| context4_multiline_v7_test_archive | 8 | 0 | 24 |
| context4_v7_test_lookup | 32 | 1 | 0 |
| context4_v7_test_pair | 45 | 7 | 19 |
| context4_v7_test_suffix | 0 | 1 | 32 |

The mixed-language trailing explanation family still has zero exact recall. Variable multiline archive material has only 8/32 exact hits. These failures remain unresolved. After inspecting them, subsequent tuning against these examples must treat this test as development and use another independently authored hold-out for fresh claims.

Historical v4 development recall regresses: 150 to 143 TP while FP falls from 58 to 37. Within that set, public-authentication audit positives regress from 22 to 5 TP; gains on HTTP transcripts (17 to 24 TP) and database credentials (42 to 45 TP) partially conceal this regression in the aggregate. The multiline tool family still has zero exact hits. Negative-challenge FP remains seven. Therefore v7 is a validated experimental candidate, not an unconditional production replacement; retain the v6 checkpoint and require broader audit-log/multiline evaluation before adoption. No production detector was switched.

Validation selected the boundary candidate before the comparison script evaluated test/development inputs. The training pipeline also emits a frozen raw-model test report after training, but that report was not used to choose the boundary head, epoch or threshold. All published numbers use saved checkpoint thresholds.
