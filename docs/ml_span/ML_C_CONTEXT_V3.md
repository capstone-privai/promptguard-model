# Context refinement v3

Additional training starts from `runs/span-mmbert-refine-v1/model`. Only prior training rows are replayed. Previously examined evaluation sets remain development sets, excluded from training.

New independently authored synthetic context families: eight train, four validation, four final test. These assignments were fixed before training. Paired hard negatives reuse the same generated strings as positives in public-reference contexts. Public evaluation value/template exclusion hashes are applied. This is synthetic context expansion, not a claim that these are verified public Rule-miss categories.

Train: 1,320 records. Validation: 256 records / four families. Test: 256 records / four unseen families. Learning rate 5e-6, two epochs, CPU four threads. Checkpoint/threshold selection uses validation F2 only. Previous-test comparisons are development regression checks.

Reproduce from the repository root:

```sh
PYTHONPATH=src .venv/bin/python scripts/prepare_context_v3.py
PYTHONPATH=src .venv/bin/python scripts/span_pipeline.py train --config configs/span-mmbert-context-v3.yaml
PYTHONPATH=src .venv/bin/python scripts/compare_context_v3.py
```

Output directories must be new. Model weights and dataset JSONL remain local and ignored by Git. No actual credential values are used. Repeated tuning after examining this test would turn it into development data; another unseen family set would then be required.

The inherited tokenizer emits a Transformers regex compatibility warning. Tokenization was preserved for checkpoint compatibility; no claim is made that this warning has been resolved. Synthetic boundary scores do not establish generalization to real credentials or public PromptGuard evaluation.

## Results

Fresh synthetic hold-out: 256 records, 128 gold spans across four families.

| Detector | Exact TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Rule | 32 | 0 | 96 | 100% | 25% |
| Previous ML | 127 | 35 | 1 | 78.40% | 99.22% |
| New ML | 128 | 0 | 0 | 100% | 100% |
| Rule + new ML candidate union | 128 | 0 | 0 | 100% | 100% |

New ML adds 96 true positives to Rule with zero induced false positives. Relative to previous ML, one additional true positive and 35 fewer false positives. Rule-miss recall: 96/96. Sequential comparison ML mean latency: previous 20.56 ms/input, new 20.44 ms/input; this small difference is not evidence of a speed improvement.

Development regression: prior refinement set remains TP48/FP0/FN0; original development set remains TP32/FP0/FN8. The eight original ML misses remain unresolved. These are already examined development sets, not independent confirmations.

Span suite: 20 passed. A stale smoke-test fixed record-count assumption was replaced by the actual generated test row count. No public evaluation performance is claimed.

See `reports/ml_span/context-v3-comparison.json`, `context-v3-evaluation.json`, `context-v3-history.json`, and `context-v3-manifest.json`. Local checkpoint: `runs/span-mmbert-context-v3/model`.
