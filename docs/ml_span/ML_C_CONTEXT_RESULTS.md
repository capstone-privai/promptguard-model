# Context refinement results

All results below are independent synthetic families or explicitly labelled development checks. This is not evaluation of the deployed PromptGuard ensemble.

Validation-selected checkpoint: `runs/span-mmbert-context-v5-boundary/model`.

## Fresh-family test

288 records; 192 gold spans; 6 families.

| Method | TP | FP | FN | Exact precision | Exact recall |
|---|---:|---:|---:|---:|---:|
| Rule | 72 | 72 | 120 | 50.00% | 37.50% |
| previous | 91 | 288 | 101 | 24.01% | 47.40% |
| candidate | 152 | 120 | 40 | 55.88% | 79.17% |
| without_boundary | 150 | 122 | 42 | 55.15% | 78.12% |
| intermediate_v4 | 152 | 128 | 40 | 54.29% | 79.17% |
| Rule + selected ML candidate union | 165 | 192 | 27 | 46.22% | 85.94% |

Selected ML adds 93 exact true positives to Rule and induces 120 false positives.
Exact Rule-miss recall: 77.50% (120 misses). No-overlap Rule-miss recall: 77.50% (120 misses). Coverage Rule-miss recall: 77.50% (120 misses).
Compared with previous ML: TP delta +61; FP delta -168.

## Family-level results

| Family | Previous TP / FP | Selected TP / FP | Gold |
|---|---:|---:|---:|
| context4_db_two_logins | 32 / 35 | 42 / 6 | 48 |
| context4_http_transcript | 4 / 22 | 17 / 7 | 24 |
| context4_ko_pair_instructions | 41 / 27 | 48 / 0 | 48 |
| context4_long_support | 7 / 49 | 21 / 26 | 24 |
| context4_multiline_tool | 2 / 110 | 0 / 57 | 24 |
| context4_public_auth_audit | 5 / 45 | 24 / 24 | 24 |

## Previously examined development sets

| Set | Previous TP / FP / FN | Selected TP / FP / FN |
|---|---:|---:|
| v3_development | 128 / 0 / 0 | 128 / 0 / 0 |
| v1_development | 48 / 0 / 0 | 48 / 0 / 0 |
| original_development | 32 / 0 / 8 | 32 / 0 / 8 |

## Unseen negative-only stress challenge

96 rows, zero gold secrets. Previous ML: 45 false-positive spans across 45 rows. Selected ML: 9 false-positive spans across 9 rows.
This set was not used for training or selection. Recall and precision with no positive findings are undefined, not 100%.

## Runtime and limitations

Sequential fresh-test mean ML latency (warm model, first scan included): previous 90.23 ms/input, selected 52.55 ms/input. Source/length mix matters; these means are not comparable to older short-input tests.
Small family counts and synthetic labels limit generalization. The boundary model only contracts existing candidates; it cannot recover absent candidates, expand truncated spans or merge fragmented multiline predictions. The local Rule proxy uses CredSweeper only and category-based file suffixes; it does not reproduce deployed per-detector ballots or exact original source-file configuration.
Validation was reused for iterative development. Test results must not be used for further tuning while retaining a hold-out claim.

Detailed machine-readable results: `reports/ml_span/context-v5-comparison.json`. Training/design: [Context v4/v5](ML_C_CONTEXT_V4.md).
