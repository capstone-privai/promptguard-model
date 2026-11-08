#!/bin/bash
# Break pilot runs' errors on model B's validation down by secret type, template origin and labelled non-secret
# kind (scripts/diagnose_pilot_step0.py), without a delimiter policy and with quote-delimiters-v2, and print each
# report as one line of JSON. Counts only.
# Usage, from the repository folder on Colab: bash scripts/colab_diag_pilot.sh RUN_NAME ...
set -euo pipefail
DRIVE=${DRIVE_DIR:-/content/drive/MyDrive/promptguard-runs}
TEAM=${TEAM_DIR:-/content/promptguard-data-pilot}/metric-detection
mkdir -p runs/pilot-reports
for name in "$@"; do
  run=${name#span-mmbert-pilot-step0-}
  for policy in none quote-delimiters-v2; do
    out=runs/pilot-reports/diag-$run-on-Bval-$policy.json
    flag=()
    [ "$policy" = none ] || flag=(--delimiter-policy "$policy")
    PYTHONPATH=src python scripts/diagnose_pilot_step0.py --checkpoint "$DRIVE/$name/model" --model B --team "$TEAM" \
      --device cuda "${flag[@]}" --output "$out" > /dev/null
    python -c "import json, sys; print(json.dumps(dict(file=sys.argv[1], report=json.load(open(sys.argv[1]))), separators=(',', ':')))" "$out"
  done
done
