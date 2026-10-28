# Context v6 results

Synthetic experiment; prior test sets are development data after the v5 error analysis. V6 uses 3,432 training rows and new 192-row validation/test sets with four families each. One epoch, batch 32, learning rate 2e-6. Model selection uses validation only.

| Dataset | Previous TP / FP / FN | V6 TP / FP / FN |
|---|---:|---:|
| validation | 46 / 106 / 50 | 47 / 26 / 49 |
| fresh_test | 61 / 124 / 35 | 54 / 20 / 42 |
| v3_development | 128 / 0 / 0 | 128 / 0 / 0 |
| v1_development | 48 / 0 / 0 | 48 / 0 / 0 |
| original_development | 32 / 0 / 8 | 40 / 0 / 0 |
| negative_challenge | 0 / 9 / 0 | 0 / 7 / 0 |
| v4_development | 152 / 120 / 40 | 150 / 58 / 42 |

Fresh-test Rule-miss recall: 0.5; ML adds 36 exact TP and 20 FP to Rule.

V6 substantially reduces false positives, but loses seven exact true positives on the fresh test and two on the earlier difficult development set. It is a precision-oriented experimental alternative, not an unqualified replacement for V5. The original eight Bearer header misses are now recovered without additional FP on that development set.

Checkpoint: `runs/span-mmbert-context-v6-boundary/model`. Boundary weights reuse the earlier synthetic-training-only model and were accepted on new validation. Public PromptGuard performance remains unmeasured.

Reproduce: run `scripts/prepare_context_v6.py`, then `scripts/span_pipeline.py train --config configs/span-mmbert-context-v6.yaml`. See `context-v6-comparison.json` for Rule, ML, union, source metrics and latency.
