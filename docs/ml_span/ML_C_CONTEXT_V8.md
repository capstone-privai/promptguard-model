# Context v8: targeted recovery and explicit regression gates

## Scope

V7 development analysis identified three unresolved weaknesses: audit-log recall regressed, a mixed-language trailing-role family had zero exact recall, and multiline file contents were usually partially predicted. At the saved v7 threshold, audit errors had 19 misses with no overlapping prediction; the multiline tool family had 23 inexact-overlap misses and one no-overlap miss; the mixed trailing-role family had 31 no-overlap misses and one inexact-overlap miss. These observations are from synthetic development cases, not additional confirmed public PromptGuard misses. They do not by themselves distinguish argmax failures from threshold removal.

The v8 generator independently authors 28 paired families: 16 training, six validation, six test, fixed before training. New training covers exposure/audit language, post-value roles in English and Korean/English, and whole multiline blocks read from synthetic paths. Negative counterparts use the same shapes and values as public documentation identifiers. Values are random synthetic strings. The 768 new training rows are combined with all 4,456 earlier training rows; historical validation/test rows remain excluded by family, template, origin and value/template hashes. New validation/test each contain 288 rows. Public value/template exclusions are retained. This does not guarantee semantic dissimilarity or cover inaccessible public data.

## Training and selection protocol

Warm-start the raw v7 model, two epochs, learning rate 1e-6, batch 32 with length grouping, CPU four threads, seed 95. Retaining historical training examples and reducing the learning rate are intended to limit forgetting; actual retention must be measured. Validation exact F2 selects the checkpoint and confidence threshold. Reuse the previously trained v2 character head only if it improves full-detector validation F2 without reducing precision. No new character training or heuristic multiline merging is introduced.

In addition to aggregate validation selection, `span-context-v8-gates.json` fixes explicit development criteria before v8 results are examined: preserve original/v1/v3 performance; restore v6 audit recall (22 exact hits with at most one FP); recover at least one exact hit in each previously zero-recall multiline-tool and mixed trailing-role family; preserve v7/v6 aggregate development results and negative-challenge FP. These are minimum checks, not a claim that one recovered hit resolves a family. Missing or undefined metrics fail the gate. Tests ensure overall improvement cannot hide an audit-family failure. Fresh-test metrics do not define these thresholds. Passing is not production certification; failing blocks recommending this candidate as a general replacement.

## Reproduction

```sh
.venv/bin/python scripts/prepare_context_v8.py
.venv/bin/python scripts/span_pipeline.py train --config configs/span-mmbert-context-v8.yaml
.venv/bin/python scripts/validate_span_boundaries.py --checkpoint runs/span-mmbert-context-v8/model --data-dir data/span-context-v8 --output runs/span-mmbert-context-v8-boundary/model --boundary-dir runs/span-boundary-v2 --report reports/ml_span/context-v8-boundary-validation.json
.venv/bin/python scripts/calibrate_span_precision.py --reference runs/span-mmbert-context-v7-boundary/model --checkpoint runs/span-mmbert-context-v8-boundary/model --data-dir data/span-context-v8 --output runs/span-mmbert-context-v8-calibrated/model --report reports/ml_span/context-v8-calibration.json
.venv/bin/python scripts/compare_span_checkpoints.py --previous runs/span-mmbert-context-v7-boundary/model --candidate runs/span-mmbert-context-v8-calibrated/model --unrefined runs/span-mmbert-context-v8/model --data-dir data/span-context-v8 --output reports/ml_span/context-v8-comparison.json --development v4_development=data/span-context-v4/test.jsonl --development v6_development=data/span-context-v6/test.jsonl --development v7_development=data/span-context-v7/test.jsonl
.venv/bin/python scripts/check_span_regressions.py --report reports/ml_span/context-v8-comparison.json --gates configs/span-context-v8-gates.json --output reports/ml_span/context-v8-regression.json
```

If the boundary gate rejects the head, compare the raw checkpoint as candidate instead. Existing output directories are not overwritten. Model weights and dataset records remain local; committed reports contain aggregates and error IDs/coordinates only. The Rule baseline is still the local proxy, not the deployed PromptGuard ensemble. All earlier tests are development data; the new six-family test is limited synthetic evidence.

## Precision-preserving validation calibration

After two epochs, the selected raw checkpoint at threshold 0.9 had 104 TP / 21 FP / 40 FN. The reused boundary head improved it to 106 TP / 19 FP / 38 FN, but precision (84.80%) was lower than v7's 86.27% on the same validation inputs. The candidate therefore was not accepted at that threshold. Before inspecting fresh-test results, validation-only calibration searched the existing six configured thresholds and required precision, recall and F2 to be at least the reference values. Threshold 0.95 was selected: 105 TP / 14 FP / 39 FN (precision 88.24%, recall 72.92%), versus v7 88 TP / 14 FP / 56 FN (86.27%, 61.11%). Threshold 0.98 also passed but had lower F2. The original raw and boundary bundles remain preserved, and the calibrated bundle records the changed threshold and its reference in metadata.

The comparison's `without_boundary` entry retains the raw model's native threshold 0.9; it differs in both boundary refinement and threshold from the final candidate, so it is not an isolated boundary ablation. The separate boundary validation report measures the head at the same threshold. Tests cover rejection of precision gains that lose reference recall, and selection when filtering a false positive preserves recall.

## Frozen evaluation results

Exact-span TP / FP / FN; historical sets are development data.

| Dataset | V7 | V8 |
|---|---:|---:|
| validation | 88 / 14 / 56 | 105 / 14 / 39 |
| fresh_test | 60 / 24 / 84 | 120 / 0 / 24 |
| v3_development | 128 / 0 / 0 | 128 / 0 / 0 |
| v1_development | 48 / 0 / 0 | 48 / 0 / 0 |
| original_development | 40 / 0 / 0 | 40 / 0 / 0 |
| negative_challenge | 0 / 7 / 0 | 0 / 0 / 0 |
| v4_development | 143 / 37 / 49 | 167 / 13 / 25 |
| v6_development | 67 / 3 / 29 | 69 / 3 / 27 |
| v7_development | 85 / 9 / 75 | 133 / 18 / 27 |

### Fresh synthetic hold-out

| Detector | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| rule_only | 0 | 48 | 144 | 0.00% | 0.00% |
| ml_only | 120 | 0 | 24 | 100.00% | 83.33% |
| candidate_union | 120 | 48 | 24 | 71.43% | 83.33% |

Exact Rule misses: 144; ML adds 120 exact TPs and 0 FPs. Importantly, Rule had no overlapping candidate for 120 gold spans: ML recovers 80.00% of these genuine no-overlap misses. Exact-boundary repairs are not all newly discovered secrets.
Mean ML latency 34.94 ms/input, p95 68.07 ms/input; CPU four threads, sequential local inference. Rule timings are separate in the JSON. Combined end-to-end deployment latency is not measured.

The fresh challenge is deliberately concentrated on contextual weaknesses. Rule has zero exact hits here, so it is not a representative all-traffic evaluation and does not replace the retained Rule-hit development checks. Rule also emits false positives which candidate union preserves; ML-only precision must not be presented as union precision. No deployed PromptGuard benchmark performance is claimed.
