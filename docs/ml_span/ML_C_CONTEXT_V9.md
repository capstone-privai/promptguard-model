# Context v9: focused audit recovery after v8 development checks

V8 improved multiline tool development TP from 0 to 24 and mixed trailing-role TP from 0 to 20, but audit TP stayed at five. Its v7 development FP also increased from nine to 18: the new suffix-family errors were inexact spans on positives, not false classification of the public values. V8 failed the predeclared development gates, so it was not treated as a general replacement.

Further development-only inspection of v8 audit positives at threshold zero showed most missing values still had no candidate; lowering a confidence threshold could not recover them. V9 therefore adds eight independent audit-training families (384 paired rows) and bounded replay of every old training family (up to the first 16 rows per family, 1,672 rows across 110 families). All new semantic pairs are retained. Total training is 2,056 rows: 984 positives and 1,072 hard negatives. Two new audit-validation and two new audit-test families are added to the v8 splits, resulting in 384 rows per evaluation split. New values and templates are synthetic and independent; all earlier evaluation groups remain outside training.

The v8 test was computed by tooling, but its metrics/examples were not inspected to design v9. Its unchanged held-out records remain in the final v9 test, alongside the two newly authored audit families. All v4/v6/v7 results used for diagnosis remain development-only. V9 warm-starts raw v8, two epochs, batch 32, learning rate 1e-6, seed 96. A new boundary-v3 model was separately trained on v8 training rows only, never on v8/v9 evaluation rows. It completed 40 epochs and selected epoch 1 with 282/288 correct character boundaries and no wrong trims on v8 validation. Full-detector validation gating is still required.

The v8 failure evidence and thresholds are preserved. Passing aggregate validation does not supersede a failed development gate, and no production detector is switched automatically. The final comparison must include v7 as the baseline plus all historical development checks. The original predeclared v8 gates are reused without weakening their bounds.

## Validation calibration and selection

Epoch 1 was retained: 156 TP / 34 FP / 36 FN before boundary correction. Epoch 2 had the same TP but two more FP. The v3 character head improved epoch 1 to 159 TP / 31 FP / 33 FN at threshold 0.95. This was insufficient to preserve v8 precision. Neither the original six-point grid nor the high-confidence five-point grid met v8's validation precision and recall together. To avoid missing feasible operating points between grid values, the final calibration evaluated all 234 observed validation confidence cutoffs. It selected 0.9996943789993322: 151 TP / 17 FP / 41 FN, versus v8 150 TP / 17 FP / 42 FN. This is only one extra validation TP and is not strong evidence of general improvement. This finely selected, very high cutoff may be sensitive to distribution or numerical changes; it needs independent evaluation rather than being treated as a calibrated probability.

All calibration attempts and their failed/accepted results are preserved. No fresh-test results were inspected before this final threshold was frozen. The final comparison uses v8 as `previous`, v9 as `candidate`, and v7 as an additional named checkpoint. Older experiments are retained rather than overwritten.

```sh
.venv/bin/python scripts/prepare_context_v9.py
.venv/bin/python scripts/span_pipeline.py train --config configs/span-mmbert-context-v9.yaml
.venv/bin/python scripts/train_span_boundaries.py --output runs/span-boundary-v3 --data-dir data/span-context-v8 --tokenizer runs/span-mmbert-context-v8/model
.venv/bin/python scripts/validate_span_boundaries.py --checkpoint runs/span-mmbert-context-v9/model --data-dir data/span-context-v9 --output runs/span-mmbert-context-v9-boundary/model --boundary-dir runs/span-boundary-v3 --report reports/ml_span/context-v9-boundary-validation.json
.venv/bin/python scripts/calibrate_span_precision.py --reference runs/span-mmbert-context-v8-calibrated/model --checkpoint runs/span-mmbert-context-v9-boundary/model --data-dir data/span-context-v9 --output runs/span-mmbert-context-v9-calibrated/model --report reports/ml_span/context-v9-exact-calibration.json --observed-thresholds
.venv/bin/python scripts/compare_span_checkpoints.py --previous runs/span-mmbert-context-v8-calibrated/model --candidate runs/span-mmbert-context-v9-calibrated/model --additional v7=runs/span-mmbert-context-v7-boundary/model --data-dir data/span-context-v9 --output reports/ml_span/context-v9-comparison.json --development v4_development=data/span-context-v4/test.jsonl --development v6_development=data/span-context-v6/test.jsonl --development v7_development=data/span-context-v7/test.jsonl
.venv/bin/python scripts/check_span_regressions.py --report reports/ml_span/context-v9-comparison.json --gates configs/span-context-v8-gates.json --output reports/ml_span/context-v9-regression.json
```

## Frozen evaluation results

Exact-span TP / FP / FN; historical sets are development data.

| Dataset | V7 | V8 | V9 |
|---|---:|---:|---:|
| validation | 133 / 17 / 59 | 150 / 17 / 42 | 151 / 17 / 41 |
| fresh_test | 73 / 27 / 119 | 148 / 1 / 44 | 159 / 1 / 33 |
| v3_development | 128 / 0 / 0 | 128 / 0 / 0 | 128 / 0 / 0 |
| v1_development | 48 / 0 / 0 | 48 / 0 / 0 | 48 / 0 / 0 |
| original_development | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 |
| negative_challenge | 0 / 7 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| v4_development | 143 / 37 / 49 | 167 / 13 / 25 | 168 / 19 / 24 |
| v6_development | 67 / 3 / 29 | 69 / 3 / 27 | 68 / 3 / 28 |
| v7_development | 85 / 9 / 75 | 133 / 18 / 27 | 122 / 14 / 38 |

### Fresh synthetic hold-out

| Detector | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| rule_only | 0 | 96 | 192 | 0.00% | 0.00% |
| ml_only | 159 | 1 | 33 | 99.38% | 82.81% |
| candidate_union | 159 | 97 | 33 | 62.11% | 82.81% |

Exact Rule misses: 192; ML adds 159 exact TPs and 1 FPs. Importantly, Rule had no overlapping candidate for 168 gold spans: ML recovers 80.36% of these genuine no-overlap misses. Exact-boundary repairs are not all newly discovered secrets.
Mean ML latency 33.10 ms/input, p95 61.08 ms/input; CPU four threads, sequential local inference. Rule timings are separate in the JSON. Combined end-to-end deployment latency is not measured.

The fresh challenge is deliberately concentrated on contextual weaknesses. Rule has zero exact hits here, so it is not a representative all-traffic evaluation and does not replace the retained Rule-hit development checks. Rule also emits false positives which candidate union preserves; ML-only precision must not be presented as union precision. No deployed PromptGuard benchmark performance is claimed.

### Predeclared development gates

| Gate | Passed | Observed |
|---|---|---|
| original retention | True | {'tp': 40, 'fp': 0} |
| v1 retention | True | {'tp': 48, 'fp': 0} |
| v3 retention | True | {'tp': 128, 'fp': 0} |
| audit recall restoration | False | {'tp': 15, 'fp': 0} |
| multiline tool recovery | True | {'tp': 23, 'fp': 0} |
| mixed trailing-role recovery | False | {'tp': 20, 'fp': 4} |
| v7 retention | False | {'tp': 122, 'fp': 14} |
| v6 retention | True | {'tp': 68, 'fp': 3} |
| negative challenge retention | True | {'fp': 0} |

Overall gate: **FAILED (6/9 passed)**. No production replacement is recommended or applied. V9 improves several targeted weaknesses over v7, but audit recall is 15/24 (below the predeclared 22 target), the older mixed trailing-role family has 20 TP with four inexact-span FP (above one), and v7 development FP is 14 (above nine). V9 also trades away some v8 gains: v4 TP 167→168 while FP 13→19; v6 TP 69→68; v7 TP 133→122 while FP 18→14. Neither v8 nor v9 is universally better across all sets.

On the new hold-out, `context4_v8_test_suffix` still has 0/24 exact recall. The new audit-item family has 16/24; mixed multiline material 23/24. Other new families have 24/24. These failures are preserved in the full report. Future tuning against them must label this test as development and introduce another independent test.

On this hold-out, 168/192 gold spans had no overlap with any Rule result; v9 correctly detects 135 of them (80.36%). The other 24 gold spans were already covered by an inexact Rule span. Thus the 159 exact additional TPs consist of 135 new no-overlap discoveries plus 24 exact-boundary recoveries. The union has 97 FP, of which 96 already came from Rule and one is added by ML. Its precision is 62.11%, despite ML-only precision of 99.38%.

The results, checkpoint hashes, failed calibrations and failed gates are retained for reproducibility. Code tests: 106 passed. Weights and generated datasets remain local and excluded from Git.

### Sequential local end-to-end timing

A separate benchmark after training/testing repeated inference and verified every candidate-union output against separately computed predictions. Its evaluation counts exactly match the frozen comparison. Model loading and external transport are excluded.

| Path | Mean ms/input | p95 ms/input |
|---|---:|---:|
| rule_latency | 0.72 | 0.69 |
| ml_latency | 32.64 | 62.73 |
| union_latency | 33.55 | 61.61 |

Reproduce: `.venv/bin/python scripts/benchmark_span_union.py --checkpoint runs/span-mmbert-context-v9-calibrated/model --data data/span-context-v9/test.jsonl --output reports/ml_span/context-v9-runtime.json`. Models are preloaded; standalone passes include their first scan and the union pass uses warmed models. This is local CPU timing, not deployed service latency.
