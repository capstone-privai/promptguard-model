#!/bin/bash
# Score the quote-delimiter policies for pilot runs trained on Colab, then print each run's record as one line of
# JSON (scripts/run_summary.py) so it reaches the local report without downloading weights.
# Usage, from the repository folder: bash scripts/colab_verify_pilot.sh RUN_NAME ...
#   RUN_NAME is a folder under DRIVE_DIR: span-mmbert-pilot-step0-<a|b|bhn>[-seedN]. B+HN keeps B's validation.
set -euo pipefail
DRIVE=${DRIVE_DIR:-/content/drive/MyDrive/promptguard-runs}
DEV="fresh_test=data/span-context-v9/test.jsonl negative_challenge=data/span-negative-challenge-v1/test.jsonl
v4=data/span-context-v4/test.jsonl v6=data/span-context-v6/test.jsonl v7=data/span-context-v7/test.jsonl
qwen_v2=data/span-qwen-holdout-v2/test.jsonl holdout_v11=data/span-holdout-v11/test.jsonl"
mkdir -p runs/pilot-reports
for name in "$@"; do
  run=${name#span-mmbert-pilot-step0-}
  model=${run%%-*}
  other=$([ "$model" = a ] && echo b || echo a)
  reports=()
  for pair in delimiters:quote-delimiters-v1 delimiters-v2:quote-delimiters-v2; do
    out=runs/pilot-reports/${pair%%:*}-pilot-$run.json
    PYTHONPATH=src python scripts/verify_span_delimiters.py --checkpoint "$DRIVE/$name/model" \
      --validation "runs/data/span-pilot-step0-$model/validation.jsonl" \
      --development "pilot_${other}_val=runs/data/span-pilot-step0-$other/validation.jsonl" $DEV \
      --device cuda --policy "${pair##*:}" --output "$out" > "$DRIVE/$name.${pair%%:*}.log" 2>&1
    reports+=("${pair%%:*}-pilot-$run=$out")
  done
  python scripts/run_summary.py export "$DRIVE/$name" "$DRIVE/$name.log" "${reports[@]}" | tee "$DRIVE/$name.summary.json"
done
